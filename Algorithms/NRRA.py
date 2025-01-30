import numpy as np

class WanderingAgent:

    def __init__(self, world: np.ndarray, movement_length: float, number_of_actions: int, consecutive_movements = None, seed = 0, agent_is_cleaner: bool = False):
        
        self.world = world
        self.move_length = movement_length
        self.number_of_actions = number_of_actions
        self.consecutive_movements = consecutive_movements
        self.t = 0
        self.action = None
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.agent_is_cleaner = agent_is_cleaner
    
    def move(self, actual_position, trash_in_pixel: bool):
        
        if trash_in_pixel and self.agent_is_cleaner and self.number_of_actions > 8:
            return 9
        else: 
            if self.action is None:
                self.action = self.select_action_without_collision(actual_position)
            
            # Compute if there is an obstacle or reached the border #
            OBS = not self.is_reachable(self.action, actual_position)

            if OBS:
                self.action = self.select_action_without_collision(actual_position)

            if self.consecutive_movements is not None:
                if self.t == self.consecutive_movements:
                    self.action = self.select_action_without_collision(actual_position)
                    self.t = 0

            self.t += 1

            return self.action
    
    
    def action_to_vector(self, action):
        """ Transform an action to a vector """

        vectors = np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)])

        return np.round(vectors[action]).astype(int)
    
    def opposite_action(self, action):
        """ Compute the opposite action """
        return (action + self.number_of_actions//2) % self.number_of_actions
    
    # def check_collision(self, action, actual_position):
    #     """ Check if the agent collides with an obstacle """
    #     new_position = actual_position + self.action_to_vector(action) * self.move_length
    #     new_position = np.round(new_position).astype(int)
        
    #     OBS = (new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
    #     if not OBS:
    #         OBS = self.world[new_position[0], new_position[1]] == 0

    #     return OBS
    
    def is_reachable(self, action, current_position):
        """ Check if the there is a path between the current position and the next position. """
        next_position = current_position + self.action_to_vector(action) * self.move_length
        next_position = np.round(next_position).astype(int)

        if self.world[int(next_position[0]), int(next_position[1])] == 0:
            return False 
        x, y = next_position
        dx = x - current_position[0]
        dy = y - current_position[1]
        steps = max(abs(dx), abs(dy))
        dx = dx / steps if steps != 0 else 0
        dy = dy / steps if steps != 0 else 0
        reachable = True
        for step in range(1, steps + 1):
            px = round(current_position[0] + dx * step)
            py = round(current_position[1] + dy * step)
            if self.world[px, py] == 0:
                reachable = False
                break

        return reachable

    def select_action_without_collision(self, actual_position):
        """ Select an action without collision """
        action_caused_collision = [not self.is_reachable(action, actual_position) for action in range(self.number_of_actions)]

        # Select a random action without collision and that is not the oppositve previous action #
        if self.action is not None:
            opposite_action = self.opposite_action(self.action)
            action_caused_collision[opposite_action] = True
        try:
            action = self.rng.choice(np.where(np.logical_not(action_caused_collision))[0])
            # action = np.random.choice(np.where(np.logical_not(action_caused_collision))[0])
        except:
            # action = np.random.randint(self.number_of_actions)
            action = opposite_action

        return action
    
    def reset(self, navigation_map):
        self.world = navigation_map