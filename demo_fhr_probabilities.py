"""
Enhanced FHR Analysis with Probability Scores and Risk Assessment
"""

print("=" * 80)
print("ENHANCED FHR ANALYSIS - NOW WITH PROBABILITY SCORES!")
print("=" * 80)

print("\n🎯 New Features Added:")
print("-" * 80)
print("""
✅ Probability/Chance Scores:
   • normal_chance: Likelihood of normal heart rate (0-100%)
   • bradycardia_chance: Risk probability of low heart rate (0-100%)
     ⚠️ Only shown if FHR is 20+ bpm below normal range
   • tachycardia_chance: Risk probability of high heart rate (0-100%)
     ⚠️ Only shown if FHR is 20+ bpm above normal range
   
   Note: Minor deviations (<20 bpm) show 100% normal_chance with 0% risk

✅ Enhanced Severity Analysis:
   • severity_level: Numeric severity (0=none, 1=mild, 2=moderate, 3=severe)
   • severity_description: Detailed description of condition
   • urgency: Specific urgency recommendations
   • risk_level: Overall risk classification

✅ Detailed Deviation Metrics:
   • deviation: Absolute deviation in bpm
   • deviation_percentage: Percentage deviation from threshold
""")

print("\n\n📊 Example API Response - NORMAL CASE:")
print("-" * 80)
print("""
Gestation: 24 weeks, Measured FHR: 145 bpm (within normal range)

{
  "fhr_analysis": {
    "gestation_weeks": 24,
    "normal_range_min": 120,
    "normal_range_max": 160,
    "measured_fhr": 145.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 100.0,         // 100% normal (within range)
    "bradycardia_chance": 0.0,      // 0% risk (no significant deviation)
    "tachycardia_chance": 0.0,      // 0% risk (no significant deviation)
    
    // STATUS
    "fhr_status": "normal",
    "fhr_classification": "Normal Heart Rate",
    
    // SEVERITY
    "severity": "none",
    "severity_level": 0,
    "severity_description": "Within normal range",
    "urgency": "Continue routine monitoring",
    "risk_level": "Low",
    
    // METRICS
    "deviation": 0,
    "deviation_percentage": 0.0,
    "medical_concern": "No"
  }
}
""")

print("\n\n⚠️ Example - MODERATE BRADYCARDIA:")
print("-" * 80)
print("""
Gestation: 24 weeks, Measured FHR: 95 bpm (25 bpm below minimum)

{
  "fhr_analysis": {
    "measured_fhr": 95.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 33.33,         // Some normal chance remains
    "bradycardia_chance": 66.67,    // 66.67% bradycardia risk (deviation > 20)
    "tachycardia_chance": 0.0,      // 0% tachycardia risk
    
    // STATUS
    "fhr_status": "bradycardia",
    "fhr_classification": "Bradycardia (Low Heart Rate)",
    
    // SEVERITY
    "severity": "moderate",
    "severity_level": 2,
    "severity_description": "Significant deviation from normal range",
    "urgency": "Medical consultation recommended soon",
    "risk_level": "Moderate to High",
    
    // METRICS
    "deviation": -25,
    "deviation_percentage": -20.83,
    "medical_concern": "Yes - Heart rate below normal range",
    "recommendation": "Immediate medical consultation recommended..."
  }
}
""")

print("\n\n⚠️ Example - MODERATE TACHYCARDIA:")
print("-" * 80)
print("""
Gestation: 24 weeks, Measured FHR: 185 bpm (25 bpm above maximum)

{
  "fhr_analysis": {
    "measured_fhr": 185.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 0.0,           // 0% normal
    "bradycardia_chance": 0.0,      // 0% bradycardia risk
    "tachycardia_chance": 83.33,    // 83.33% tachycardia risk
    
    // STATUS
    "fhr_status": "tachycardia",
    "fhr_classification": "Tachycardia (High Heart Rate)",
    
    // SEVERITY
    "severity": "moderate",
    "severity_level": 2,
    "severity_description": "Significant deviation from normal range",
    "urgency": "Medical consultation recommended soon",
    "risk_level": "Moderate to High",
    
    // METRICS
    "deviation": +25,
    "deviation_percentage": +15.63,
    "medical_concern": "Yes - Heart rate above normal range"
  }
}
""")

print("\n\n🚨 Example - SEVERE BRADYCARDIA:")
print("-" * 80)
print("""
Gestation: 24 weeks, Measured FHR: 70 bpm (50 bpm below minimum)

{
  "fhr_analysis": {
    "measured_fhr": 70.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 0.0,           // 0% normal
    "bradycardia_chance": 100.0,    // 100% bradycardia - CRITICAL
    "tachycardia_chance": 0.0,      // 0% tachycardia risk
    
    // STATUS
    "fhr_status": "bradycardia",
    "fhr_classification": "Bradycardia (Low Heart Rate)",
    
    // SEVERITY
    "severity": "severe",
    "severity_level": 3,
    "severity_description": "Critical deviation from normal range",
    "urgency": "Immediate medical attention required",
    "risk_level": "High to Critical",
    
    // METRICS
    "deviation": -50,
    "deviation_percentage": -41.67,
    "medical_concern": "Yes - Critical situation"
  }
}
""")

print("\n\n📈 Severity Level Guide:")
print("-" * 80)
print("""
Level 0 (None):     Normal range - Continue routine care
Level 1 (Mild):     Slight deviation - Monitor closely, consult if persists
Level 2 (Moderate): Significant deviation - Medical consultation soon
Level 3 (Severe):   Critical deviation - Immediate medical attention
""")

print("\n\n🔍 How Probabilities Are Calculated:")
print("-" * 80)
print("""
The system uses intelligent algorithms to calculate risk probabilities:

1. Within Normal Range OR minor deviation (<20 bpm):
   - normal_chance: 100%
   - bradycardia_chance: 0%
   - tachycardia_chance: 0%

2. Below Normal by 20+ bpm (Bradycardia):
   - bradycardia_chance: 50-100% (increases with deviation beyond 20 bpm)
   - normal_chance: 0-50%
   - tachycardia_chance: 0%

3. Above Normal by 20+ bpm (Tachycardia):
   - tachycardia_chance: 50-100% (increases with deviation beyond 20 bpm)
   - normal_chance: 0-50%
   - bradycardia_chance: 0%

All probabilities always sum to 100%

Example: FHR = 115 at 24 weeks (normal range: 120-160)
  Deviation = -5 bpm (less than 20)
  Result: normal_chance = 100%, bradycardia_chance = 0%, tachycardia_chance = 0%

Example: FHR = 95 at 24 weeks (normal range: 120-160)
  Deviation = -25 bpm (more than 20)
  Result: normal_chance = ~20%, bradycardia_chance = ~80%, tachycardia_chance = 0%
""")

print("\n" + "=" * 80)
print("✅ Enhanced FHR Analysis Ready for Deployment!")
print("=" * 80)
print("\nCommit and push to get these features on Railway:")
print("  git add .")
print("  git commit -m 'Add probability scores and enhanced risk assessment'")
print("  git push origin main")
print("=" * 80)
937899, 'abnormal_beats': 2, 'mean_fhr': 148.77120277735165, 'median_fhr': 167.13091922005654, 'min_fhr': 76.77543186180428, 'max_fhr': 200.0000000000001, 'fhr_range': 123.22456813819583, 'short_term_variability': 41.5964002610865}, 'gestation_period': '24 weeks', 'original_filename': 'a0001.wav', 'status': 'abnormal', 'message': 'Potential abnormality detected in fetal heart sounds.', 'recommendation': 'Please consult your healthcare provider immediately for further evaluation.'}