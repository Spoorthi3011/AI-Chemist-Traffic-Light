class AIChemist:
    def __init__(self):
        self.performance_history = []
        self.base_params = {'OXIDATION': 1200, 'REDUCTION': 1500}
    
    def recommend_rpm(self, phase):
        """AI initial recommendation"""
        print(f"🧠 AI Chemist: Phase {phase} → {self.base_params[phase]} RPM")
        return self.base_params[phase]
    
    def adapt_rpm(self, previous_rpm, transition_speed):
        """🆕 Novel: Real-time stirring optimization"""
        if transition_speed < 0.1:  # Slow reaction
            new_rpm = min(previous_rpm + 200, 2000)
            rationale = "accelerating slow reaction"
        elif transition_speed > 0.4:  # Too violent
            new_rpm = max(previous_rpm - 100, 800)
            rationale = "calming violent reaction"
        else:
            new_rpm = previous_rpm
            rationale = "optimal kinetics"
        
        print(f"🧠 AI ADAPT: {previous_rpm}→{new_rpm} RPM ({rationale}, speed={transition_speed:.2f}px/s)")
        self.performance_history.append({'rpm': new_rpm, 'speed': transition_speed})
        return new_rpm