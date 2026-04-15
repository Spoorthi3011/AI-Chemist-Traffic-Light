import matplotlib.pyplot as plt
import numpy as np
import time
import threading

class DigitalTwin:
    def __init__(self):
        self.history = {'GREEN': [], 'RED': [], 'YELLOW': []}
        self.timestamps = []
        self._lock = threading.Lock()
        
        # Thread-safe matplotlib (non-interactive)
        plt.ioff()  # Disable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.title('🧪 LIVE AI-CHEMIST Digital Twin')
        print("✅ DigitalTwin: Thread-safe mode")
    
    def update(self, detector):
        """Thread-safe live updates (no GUI in thread)"""
        with self._lock:
            now = time.time()
            self.timestamps.append(now)
            
            for color in ['GREEN', 'RED', 'YELLOW']:
                count = detector.pixel_counts.get(color, 0)
                self.history[color].append(count)
            
            # Trim to 200 points
            trim = max(0, len(self.timestamps) - 200)
            self.timestamps = self.timestamps[trim:]
            for color in self.history:
                self.history[color] = self.history[color][trim:]
    
    def render(self):
        """Call this from MAIN THREAD only for live plot"""
        with self._lock:
            if len(self.timestamps) < 2:
                return
            
            self.ax.clear()
            colors = {'GREEN': 'lime', 'RED': 'red', 'YELLOW': 'gold'}
            
            for color, data in self.history.items():
                if len(data) > 0:
                    t = np.array(self.timestamps) - self.timestamps[0]
                    self.ax.plot(t, data, label=color, linewidth=4, color=colors[color])
            
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Pixel Intensity')
            self.ax.legend(fontsize=12)
            self.ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
    
    def save_report(self, filename):
        """High-quality final report"""
        with self._lock:
            if len(self.timestamps) == 0:
                print("⚠️  No data for report")
                return
            
            self.ax.clear()
            colors = {'GREEN': 'lime', 'RED': 'red', 'YELLOW': 'gold'}
            
            for color, data in self.history.items():
                if len(data) > 0:
                    t = np.array(self.timestamps) - self.timestamps[0]
                    self.ax.plot(t, data, label=color, linewidth=4, color=colors[color])
            
            self.ax.set_title('🏆 AI-CHEMIST Complete Reaction Profile')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Pixel Intensity')
            self.ax.legend(fontsize=12)
            self.ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Digital Twin saved: {filename}")
