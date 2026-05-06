import time
import numpy as np


class ReactionAnalyzer:
    def __init__(self):
        self.phase_metrics = []

    def analyze_phase(self, detector):
        """
        Quantitative chemistry metrics
        """

        # Safe handling for missing state_start_time
        if not hasattr(detector, "state_start_time"):
            detector.state_start_time = time.time()

        duration = time.time() - detector.state_start_time

        # Safe handling for missing pixel_counts
        if not hasattr(detector, "pixel_counts"):
            detector.pixel_counts = {
                'RED': 0,
                'GREEN': 0,
                'YELLOW': 0
            }

        pixels = detector.pixel_counts

        total_pixels = sum(pixels.values())

        if total_pixels > 0:
            dominant_color = max(pixels, key=pixels.get)
            yield_pct = (pixels[dominant_color] / total_pixels) * 100
        else:
            dominant_color = "UNKNOWN"
            yield_pct = 0

        # Transition speed metrics
        red_speed = pixels.get('RED', 0) / max(duration, 1)
        yellow_speed = pixels.get('YELLOW', 0) / max(duration, 1)

        metrics = {
            'yield_%': f"{yield_pct:.1f}%",
            'duration_s': f"{duration:.1f}s",
            'red_speed': red_speed,
            'yellow_speed': yellow_speed,
            'dominant': dominant_color
        }

        self.phase_metrics.append(metrics)

        print(f"Phase Analysis: {metrics}")

        return metrics

    def final_report(self):
        """
        Complete experiment summary
        """

        if not self.phase_metrics:
            return "No phase data"

        yields = [
            float(m['yield_%'].replace('%', ''))
            for m in self.phase_metrics
        ]

        avg_yield = np.mean(yields)

        max_red_speed = max(
            m['red_speed']
            for m in self.phase_metrics
        )

        adaptation_count = len(
            set(
                m['red_speed']
                for m in self.phase_metrics
            )
        )

        report = f"""
FINAL CHEMISTRY REPORT
======================

Average Yield: {avg_yield:.1f}%
Phases Complete: {len(self.phase_metrics)}
Max Red Speed: {max_red_speed:.1f} px/s
AI Adaptations: {adaptation_count}
"""

        print(report)

        return report
