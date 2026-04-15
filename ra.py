import time
class ReactionAnalyzer:
    def __init__(self):
        self.phase_metrics = []
    
    def analyze_phase(self, detector):
        """🆕 Quantitative chemistry metrics"""
        duration = time.time() - detector.state_start_time
        pixels = detector.pixel_counts
        
        total_pixels = sum(pixels.values())
        dominant_color = max(pixels, key=pixels.get)
        yield_pct = (pixels[dominant_color] / total_pixels * 100) if total_pixels > 0 else 0
        
        # Novel transition speed metric
        red_speed = pixels.get('RED', 0) / max(duration, 1)
        
        metrics = {
            'yield_%': f"{yield_pct:.1f}%",
            'duration_s': f"{duration:.1f}s",
            'red_speed': red_speed,
            'dominant': dominant_color
        }
        self.phase_metrics.append(metrics)
        print(f"📊 Phase Analysis: {metrics}")
        return metrics
    
    def final_report(self):
        """Complete experiment summary"""
        if not self.phase_metrics:
            return "No phase data"
        
        yields = [float(m['yield_%'][:-1]) for m in self.phase_metrics]
        avg_yield = np.mean(yields)
        
        report = f"""
🏆 FINAL NOVEL CHEMISTRY REPORT
═══════════════════════════════
📈 Average Yield: {avg_yield:.1f}%
🔄 Phases Complete: {len(self.phase_metrics)}
⚡ Max Red Speed: {max(m['red_speed'] for m in self.phase_metrics):.1f} px/s
🎯 AI Adaptations: {len(set([m['red_speed'] for m in self.phase_metrics]))}
        """
        print(report)
        return report
