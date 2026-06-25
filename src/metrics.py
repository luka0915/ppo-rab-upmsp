import matplotlib.pyplot as plt
import numpy as np
import os

class MetricsTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rewards = []
        self.tardiness = []
        self.makespans = []
        self.episodes = []

        # Window size for moving average
        self.window_size = 100

        if not os.path.exists(self.cfg.RESULTS_DIR):
            os.makedirs(self.cfg.RESULTS_DIR)

    def get_moving_average(self, data, window_size):
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return np.mean(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def update(self, episode, reward, tardiness, makespan):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.tardiness.append(tardiness)
        self.makespans.append(makespan)

        if (episode + 1) % self.cfg.PLOT_INTERVAL == 0:
            self.save_plots()
            self.print_statistics()

    def print_statistics(self):
        """Print training statistics so far."""
        print("\nPerformance Statistics:")
        print(f"Current Episode: {self.episodes[-1]}")
        print("\nBest Performance:")
        print(f"Best MWT: {min(self.tardiness):.2f}")
        print(f"Best Reward: {max(self.rewards):.2f}")

        last_100_episodes = slice(-100, None)
        print("\nRecent Performance (Last 100 episodes):")
        print(f"Average MWT: {np.mean(self.tardiness[last_100_episodes]):.2f}")
        print(f"Average Reward: {np.mean(self.rewards[last_100_episodes]):.2f}")

    def save_plots(self):
        """Plot only episode reward and mean weighted tardiness (MWT)."""
        if not os.path.exists(self.cfg.RESULTS_DIR):
            os.makedirs(self.cfg.RESULTS_DIR)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Reward plot
        ax1.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Raw')
        if len(self.rewards) >= self.window_size:
            moving_avg = self.get_moving_average(self.rewards, self.window_size)
            ax1.plot(self.episodes[self.window_size-1:], moving_avg, 'b-',
                    label=f'Moving Average ({self.window_size})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        ax1.legend()

        # Mean Weighted Tardiness plot
        ax2.plot(self.episodes, self.tardiness, 'r-', alpha=0.3, label='Raw')
        if len(self.tardiness) >= self.window_size:
            moving_avg = self.get_moving_average(self.tardiness, self.window_size)
            ax2.plot(self.episodes[self.window_size-1:], moving_avg, 'r-',
                    label=f'Moving Average ({self.window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('MWT')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{self.cfg.RESULTS_DIR}training_plots.png")
        plt.close(fig)

    def get_best_performance(self):
        """Return the best performance."""
        best_tardiness_idx = np.argmin(self.tardiness)
        return {
            'episode': self.episodes[best_tardiness_idx],
            'reward': self.rewards[best_tardiness_idx],
            'tardiness': self.tardiness[best_tardiness_idx],
            'makespan': self.makespans[best_tardiness_idx]
        }
    
class TestMetricsTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = []
        self.tardiness = []
        self.makespans = []
        self.setup_times = []
        
        # Ensure the results directory exists
        if not os.path.exists(self.cfg.RESULTS_DIR):
            os.makedirs(self.cfg.RESULTS_DIR)
    
    def update(self, step, tardiness, makespan, setup_time):
        self.steps.append(step)
        self.tardiness.append(tardiness)
        self.makespans.append(makespan)
        self.setup_times.append(setup_time)
        self.save_plots()
    
    def save_plots(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # MWT plot - drawn as a step function
        ax1.step(self.steps, self.tardiness, 'r-', where='post', label='MWT')
        ax1.set_title('Mean Weighted Tardiness')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('MWT')
        ax1.grid(True)
        ax1.legend()
        
        # Makespan plot - drawn as a step function
        ax2.step(self.steps, self.makespans, 'b-', where='post', label='Makespan')
        ax2.set_title('Makespan')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Time')
        ax2.grid(True)
        ax2.legend()
        
        # Setup time plot - drawn as a step function
        ax3.step(self.steps, self.setup_times, 'g-', where='post', label='Setup Time')
        ax3.set_title('Total Setup Time')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Time')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.cfg.RESULTS_DIR}test_progress.png")
        plt.close()