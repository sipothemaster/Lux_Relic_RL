
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy
import numpy as np
from sys import stderr
from scipy.signal import convolve2d

from base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    get_opposite,
    is_team_sector,
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: bool):
        '''
        if self._explored_for_relic and self._relic != status:\
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )
        '''

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: bool):
        '''
        if self._explored_for_reward and self._reward != status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )
        '''

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(obs, team_id, team_reward)

    def _update_relic_map(self, obs, team_id, team_reward):
        for relic_id, (mask, xy) in enumerate(
            zip(obs["relic_nodes_mask"], obs["relic_nodes"])
        ):
            if mask:
                self._update_relic_status(*xy, status=True)

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) == Global.MAX_RELIC_NODES:
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_status_from_relics_distribution()
            self._update_reward_results(obs, team_id, team_reward)
            self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        # We will use Global.REWARD_RESULTS to identify which nodes yield points
        for result in Global.REWARD_RESULTS:

            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue

                if n.reward:
                    known_reward += 1
                    continue

                unknown_nodes.add(n)

            if not unknown_nodes:
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            elif reward > len(unknown_nodes):
                # we shouldn't be here
                print(
                    f"Something wrong with reward result: {result}"
                    ", this result will be ignored.",
                    file=stderr,
                )

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status=True):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                # Can't find OBSTACLE_MOVEMENT_DIRECTION
                for node in self:
                    node.type = NodeType.unknown

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        # Right now there are only two options for nebula_tile_drift_speed: 1 / 20 and 1 / 40
        if obstacles_movement_status and obstacles_movement_status[-1]:
            return 20 if len(obstacles_movement_status) - 21 % 40 < 20 else 40

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.OBSTACLE_MOVEMENT_PERIOD > 0
            and (step - 1) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
        ):
            self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:

    def __init__(self, player: str, env_cfg, training=False) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.training = training
        

        # DQN Parameters
        self.state_size = 13  
        self.action_size = 2  
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Initialize DQN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)

        if not training:
            self.load_model()
            self.epsilon = 0.0


        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def _simulate_movement(self, ship):
        """
        Simulate the ship's next position based on the action.
    
        Args:
            ship (Ship): The current ship.
            action (int): The action to simulate.
    
        Returns:
            tuple: (next_x, next_y) The predicted coordinates.
        """
        x, y = ship.coordinates
        action = ship.action
        if action == ActionType.up:
            return x, y - 1
        elif action == ActionType.down:
            return x, y + 1
        elif action == ActionType.left:
            return x - 1, y
        elif action == ActionType.right:
            return x + 1, y
        return x, y  # No movement for ActionType.center

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        self.current_step = get_match_step(step)

        # print(f"start step={match_step}({step})", file=stderr)

        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        # self.show_visible_map()

        self.find_relics()
        self.find_rewards()
        self.harvest()
        self.attack()     # Assign kill tasks to idle agents

        #for ship in self.fleet:
             #print(ship, ship.task, ship.target, ship.action, file=stderr)

        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                actions[i] = ship.action, 0, 0

        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:

                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in self.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < Global.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    # If one ship is stationary, we will move all the other ships.
                    # This will help generate more useful data in Global.REWARD_RESULTS.
                    can_pause = False
            else:
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def harvest(self):

        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def attack(self):
        
        def set_upgrade_task(ship, target_node):
    
            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
    
            if not actions or ship.energy < energy:
                return False
    
            ship.task = "upgrade"
            ship.target = target_node
            ship.action = actions[0]
            return True
    
        def set_kill_task(ship, target_node):
            
            if ship.node == target_node:
                ship.task = "kill"
                ship.target = target_node
                ship.action = ActionType.center  # Stay on the node
                return True
            
            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
    
            if not actions or ship.energy < energy:
                return False

            ship.task = "kill"
            ship.target = target_node
            ship.action = actions[0]
            return True

        # Prepare the top 10% energy nodes for upgrade task
        all_energy_nodes = [
            node for node in self.space if node.energy and node.is_walkable
        ]
        sorted_energy_nodes = sorted(all_energy_nodes, key=lambda n: n.energy, reverse=True)
        top_energy_nodes = sorted_energy_nodes[: max(1, len(sorted_energy_nodes) // 10)]
    
        # Prepare mirrored nodes of enemy reward nodes for kill task
        enemy_reward_nodes = [
            get_opposite(node.x, node.y) for node in self.space.reward_nodes
        ]
        potential_targets = {
            "upgrade": [(node.x, node.y) for node in top_energy_nodes],
            "kill": enemy_reward_nodes,
        }
        
        for ship in self.fleet:
            if ship.task == "kill" and ship.node == ship.target:
                target_tuple = (ship.target.x, ship.target.y) 
                if target_tuple in potential_targets["kill"]:
                    potential_targets["kill"].remove(target_tuple)
                
        
        if not potential_targets["upgrade"] or not potential_targets["kill"]: # to stand by
            #print("No valid targets for upgrade or kill tasks now")
            return

        for node in self.space:
            if not node.explored_for_relic:
                return
        if self.get_unexplored_relics():
            return
            

        for ship in self.fleet:
            
            done = False
            upgrade_target = find_closest_target(ship.coordinates, potential_targets["upgrade"])[0]
            kill_target = find_closest_target(ship.coordinates, potential_targets["kill"])[0]
            
            if ship.task == "upgrade" and ship.node == ship.target: # If reach the upgrade target
                ship.task = None # Reassign attack task (kill or upgrade)
                ship.target = None # Reassign attack target
                ship.action = None # Reassign task action
                done = True

            if ship.task:
                if ship.task == "kill" and (ship.target.x, ship.target.y) != kill_target:
                    set_kill_task(ship, self.space.get_node(*kill_target)) # find another target instead
                
                else:  # Skip ships that already have tasks
                    continue

            # Prepare RL state for the ship
            state = self._prepare_attack_state(ship, upgrade_target, kill_target)
    
            # Decide task using epsilon-greedy policy
            if random.random() < self.epsilon and self.training:
                task_type = random.choice(["upgrade", "kill"])
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    task_type = "upgrade" if q_values[0] > q_values[1] else "kill"
                    
            # Assign the chosen task
            if task_type == "upgrade" and potential_targets["upgrade"]:
                if upgrade_target and set_upgrade_task(ship, self.space.get_node(*upgrade_target)):
                    continue
    
            elif task_type == "kill" and potential_targets["kill"]:
                if kill_target and set_kill_task(ship, self.space.get_node(*kill_target)):
                    # potential_targets["kill"].remove(kill_target) # removed newly booked kill target(this would result in no sufficient agent)
                    continue
    
            # Store experience in ReplayBuffer
            if self.training:
                rl_reward = 0 
                if (ship.coordinates in potential_targets["upgrade"]):
                    rl_reward += 1
        
                if (ship.coordinates in potential_targets["kill"]) and ship.energy > 0:
                    rl_reward += 100 
                
                ship_copy = copy.deepcopy(ship)
                # Simulate the ship's movement
                if ship_copy.action == ActionType.center:
                    pass  # Ship stays in place
                elif ship_copy.action in [ActionType.up, ActionType.down, ActionType.left, ActionType.right]:
                    new_x, new_y = self._simulate_movement(ship_copy)
                    ship_copy.node = self.space.get_node(new_x, new_y)
        
                if ship_copy.node and ship_copy.node.energy:
                    ship_copy.energy += ship_copy.node.energy
                ship_copy.energy -= Global.UNIT_MOVE_COST  # Subtract move cost
            
                next_state = self._prepare_attack_state(ship_copy, upgrade_target, kill_target) # update ship coordinates here
    
                self.memory.push(state, 0 if task_type == "upgrade" else 1, rl_reward, next_state, done)
                self.train()
    
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def _prepare_attack_state(self, ship, upgrade_target, kill_target):
        """
        Prepare the state representation for RL decision-making, including cost estimation.
        """
        # Ship's position and energy
        x, y = ship.coordinates
        energy = ship.energy


        if upgrade_target:
            upgrade_x, upgrade_y = upgrade_target
            upgrade_distance = manhattan_distance(ship.coordinates, upgrade_target)
            upgrade_energy = self.space.get_node(upgrade_x, upgrade_y).energy or 0
            upgrade_path = astar(create_weights(self.space), ship.coordinates, upgrade_target)
            upgrade_cost = estimate_energy_cost(self.space, upgrade_path)
        else:
            upgrade_x, upgrade_y = -1, -1  # No valid target
            upgrade_distance = Global.SPACE_SIZE * 2
            upgrade_energy = 0
            upgrade_cost = float('inf')  # Large cost for invalid targets
    
        
        if kill_target:
            kill_x, kill_y = kill_target
            kill_distance = manhattan_distance(ship.coordinates, kill_target)
            kill_path = astar(create_weights(self.space), ship.coordinates, kill_target)
            kill_cost = estimate_energy_cost(self.space, kill_path)
        else:
            kill_x, kill_y = -1, -1  # No valid target
            kill_distance = Global.SPACE_SIZE * 2
            kill_cost = float('inf')  # Large cost for invalid targets
    
        # Normalized step
        normalized_step = self.current_step / Global.MAX_STEPS_IN_MATCH
    
        # Create the state vector
        state = np.array([
            x, y, energy,                    # Ship's position and energy
            upgrade_x, upgrade_y,            # Closest upgrade target position
            upgrade_distance, upgrade_energy, upgrade_cost,  # Upgrade target features
            kill_x, kill_y,                  # Closest kill target position
            kill_distance, kill_cost,        # Kill target features
            normalized_step                  # Normalized current step
        ])
    
        return torch.FloatTensor(state).to(self.device)




    def train(self):
        if len(self.memory) < self.batch_size:
            return  # Skip training if not enough samples
    
        # Sample a batch of experiences
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
    
        # Compute Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # Periodically update target network
        if self.current_step % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth', weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            current_directory = os.getcwd()
            print(f"Current Directory: {current_directory}")

            all_items = os.listdir(current_directory)
            print("All items in the current directory:")

            raise FileNotFoundError(f"No trained model found for {self.player}")


    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
