import numpy as np
import pygame
import pickle as pk
import gymnasium as gym
from gymnasium import spaces


class MARCPTR(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.user_id = None
        self.reward_table =  None
        self.size = 25  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(self.size)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {}
        for item in range(self.size):
            self._action_to_direction[item] = np.array([0, item])
        '''
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        '''
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": abs(self._agent_location - self._target_location)}

    def reset(self, seed=None, options=None, start=None, end=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        
        if(start==None and end==None):
            self._target_location = np.array([0, np.random.randint(self.size)])

            self._agent_location = self._target_location
            while np.array_equal(self._target_location, self._agent_location):
                self._agent_location = np.array([0, np.random.randint(self.size)])
        else:
            self._agent_location = np.array([0, start])
            self._target_location = np.array([0, end])
            


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        #print(action)
        #print(self._agent_location)
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        
        '''
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        '''
        old_location = self._agent_location
        self._agent_location = direction
        
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        #reward = 1 if terminated else 0 + self.reward_function(old_location, self._agent_location) # Binary sparse rewards
        reward = self.reward_function(old_location, self._agent_location) # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def set_size(self, size):
        self.size = size
        
        self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                }
        )

        self.action_space = spaces.Discrete(size)
        
        
        for item in range(size):
            self._action_to_direction[item] = np.array([0, item])
        return self.size
        
    def define_user(self, userid, city):
        self.user_id = userid
        
        file = open('data_process/' + city + '/adjacency', 'rb')
        data = pk.load(file)
        
        self.reward_table = data[self.user_id]
    
    def reward_function(self, old, new):
        #print(self.reward_table[old[1], new[1]])
        #print(old[1], new[1])
        #print(self.reward_table[old][new])
        
        return self.reward_table[old[1], new[1]]
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    