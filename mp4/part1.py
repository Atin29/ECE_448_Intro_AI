import random
import numpy as np

PADDLE_HEIGHT = 0.2

class agent:
    def __init__(self):
        self.score = 0

        # dict mapping from state to utility
        self.state_utility = {}

        # dict mapping from state and action to utility
        self.action_utility = {}

        # continuous values of (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        self.state = (0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT/2)

    def train():
        """Train to calculate the function values for state_utility and
        action_utility."""
        pass

    def act(self):
        """Determine the best action to take from the current state.
        Update state after the agent performs that action.
        """
        pass

    def discretize_state(self):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state
        ball_x = int(12*ball_x)
        ball_y = int(12*ball_y)
        velocity_x = 1 if velocity_x >= 0 else -1
        velocity_y = 0 if abs(velocity_y) < 0.015 else (1 if velocity_y > 0 else -1)
        paddle_y = 11 if paddle_y == 1 - PADDLE_HEIGHT else int(12*paddle_y / (1 - PADDLE_HEIGHT))
        print('discretized state:', ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        return (ball_x, ball_y, velocity_x, velocity_y, paddle_y)

    def move_paddle_up(self):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state
        paddle_y -= 0.04
        if paddle_y < 0:
            paddle_y = 0
        self.state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        return self.state

    def move_paddle_down(self):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state
        paddle_y += 0.04
        if paddle_y > 1:
            paddle_y = 1 - PADDLE_HEIGHT
        self.state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        return self.state

    def no_paddle_move(self):
        return self.state

    def update_state(self):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state

        ball_x += velocity_x
        ball_y += velocity_y

        if ball_y < 0:
            # top wall bounce
            ball_y = -ball_y
            velocity_y = -velocity_y

        if ball_y > 1:
            # bottom wall bounce
            ball_y = 2 - ball_y
            velocity_y = -velocity_y

        if ball_x < 0:
            # left wall bounce
            ball_x = -ball_x
            velocity_x = -velocity_x

        if ball_x >= 1 and paddle_y <= ball_y <= (paddle_y+PADDLE_HEIGHT):
            # paddle bounce
            ball_x = 2*paddle_x - ball_x
            velocity_x = -velocity_x + random.uniform(-0.015, 0.015)
            velocity_y = velocity_y + random.uniform(-0.03, 0.03)
        elif ball_x >= 1:
            ball_x = -1

        if velocity_x > 1:
            velocity_x = 1
        if velocity_x < -1:
            velocity_x = -1
        if velocity_y > 1:
            velocity_y = 1
        if velocity_y < -1:
            velocity_y = -1

        if 0 <= velocity_x < 0.03:
            velocity_x = 0.03
        if -0.03 < velocity_x < 0:
            velocity_x = -0.03

        self.state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        return self.state

    def calculate_state_utility(state):
        """Return U(s) = max_a Q(s,a)"""
        pass

    def calculate_action_utility(state, action):
        """Return Q(s,a) = ..."""
        pass


def main():
    a = agent()
    a.discretize_state()

if __name__ == '__main__':
    main()
