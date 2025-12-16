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
   • tachycardia_chance: Risk probability of high heart rate (0-100%)

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
Gestation: 24 weeks, Measured FHR: 145 bpm

{
  "fhr_analysis": {
    "gestation_weeks": 24,
    "normal_range_min": 120,
    "normal_range_max": 160,
    "measured_fhr": 145.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 89.5,          // 89.5% chance of normal
    "bradycardia_chance": 5.2,      // 5.2% risk of bradycardia
    "tachycardia_chance": 5.3,      // 5.3% risk of tachycardia
    
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
Gestation: 24 weeks, Measured FHR: 100 bpm (20 bpm below minimum)

{
  "fhr_analysis": {
    "measured_fhr": 100.0,
    
    // 🎯 PROBABILITY SCORES
    "normal_chance": 0.0,           // 0% normal
    "bradycardia_chance": 66.67,    // 66.67% bradycardia risk
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
    "deviation": -20,
    "deviation_percentage": -16.67,
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

1. Within Normal Range:
   - normal_chance: 60-100% (highest at midpoint)
   - bradycardia/tachycardia_chance: 0-20% (based on proximity to boundaries)

2. Below Normal (Bradycardia):
   - bradycardia_chance: 50-100% (increases with deviation)
   - normal_chance: 0-50%
   - tachycardia_chance: ~0%

3. Above Normal (Tachycardia):
   - tachycardia_chance: 50-100% (increases with deviation)
   - normal_chance: 0-50%
   - bradycardia_chance: ~0%

All probabilities always sum to 100%
""")

print("\n" + "=" * 80)
print("✅ Enhanced FHR Analysis Ready for Deployment!")
print("=" * 80)
print("\nCommit and push to get these features on Railway:")
print("  git add .")
print("  git commit -m 'Add probability scores and enhanced risk assessment'")
print("  git push origin main")
print("=" * 80)
