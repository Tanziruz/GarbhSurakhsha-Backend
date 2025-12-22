"""
Test script to verify the API server is working correctly
"""
import requests
import json
import sys
from pathlib import Path

# Default to Railway deployment, but allow local testing
if len(sys.argv) > 1 and sys.argv[1] == "local":
    BASE_URL = "http://localhost:8000"
    print("Testing LOCAL server")
else:
    BASE_URL = "https://garbhsurakhsha.up.railway.app"
    print("Testing RAILWAY deployment")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Is it running?")
        print("  Run: python api_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_analyze_audio(audio_file="a0001.wav", gestation_period="24 weeks"):
    """Test the audio analysis endpoint"""
    print(f"\nTesting audio analysis with {audio_file}...")

    # Check if audio file exists
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_file}")
        print("  Please provide a valid WAV file for testing")
        return False

    try:
        with open(audio_path, 'rb') as f:
            files = {'audio_file': (audio_file, f, 'audio/wav')}
            data = {'gestation_period': gestation_period}

            response = requests.post(
                f"{BASE_URL}/analyze",
                files=files,
                data=data,
                timeout=30
            )

        print(f"✓ Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*70)
            print("ANALYSIS RESULT - COMPLETE JSON RESPONSE")
            print("="*70)
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Status: {result['status']}")
            print(f"\nMessage: {result['message']}")
            print(f"\nRecommendation: {result['recommendation']}")

            print("="*70)
            print("="*70)

            print(result)
            
            # Probabilities
            print(f"\nProbabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.2%}")
            
            # Heart Rate Metrics
            if 'heart_rate' in result:
                print(f"\n{'='*70}")
                print("HEART RATE ANALYSIS")
                print("="*70)
                hr = result['heart_rate']
                if 'average_fhr' in hr:
                    print(f"Average FHR: {hr['average_fhr']:.1f} bpm")
                    print(f"Beat Count: {hr.get('beat_count', 'N/A')}")
                    print(f"Mean FHR: {hr.get('mean_fhr', 'N/A'):.1f} bpm")
                    print(f"Median FHR: {hr.get('median_fhr', 'N/A'):.1f} bpm")
                    print(f"Min FHR: {hr.get('min_fhr', 'N/A'):.1f} bpm")
                    print(f"Max FHR: {hr.get('max_fhr', 'N/A'):.1f} bpm")
                    print(f"FHR Range: {hr.get('fhr_range', 'N/A'):.1f} bpm")
                else:
                    print(f"Error: {hr.get('error', 'Unknown error')}")
            
            # FHR Analysis with Probabilities
            if 'fhr_analysis' in result:
                print(f"\n{'='*70}")
                print("FHR RISK ANALYSIS")
                print("="*70)
                fhr_a = result['fhr_analysis']
                
                print(f"Gestation: {fhr_a.get('gestation_weeks', 'N/A')} weeks")
                print(f"Normal Range: {fhr_a.get('normal_range_min', 'N/A')}-{fhr_a.get('normal_range_max', 'N/A')} bpm")
                print(f"Measured FHR: {fhr_a.get('measured_fhr', 'N/A'):.1f} bpm")
                
                print(f"\n🎯 PROBABILITY SCORES:")
                print(f"  Normal Chance:      {fhr_a.get('normal_chance', 0):.2f}%")
                print(f"  Bradycardia Chance: {fhr_a.get('bradycardia_chance', 0):.2f}%")
                print(f"  Tachycardia Chance: {fhr_a.get('tachycardia_chance', 0):.2f}%")
                
                print(f"\n📊 STATUS:")
                print(f"  FHR Status: {fhr_a.get('fhr_status', 'unknown').upper()}")
                print(f"  Classification: {fhr_a.get('fhr_classification', 'N/A')}")
                
                print(f"\n⚠️ SEVERITY:")
                print(f"  Severity: {fhr_a.get('severity', 'none').upper()}")
                print(f"  Severity Level: {fhr_a.get('severity_level', 0)}/3")
                print(f"  Description: {fhr_a.get('severity_description', 'N/A')}")
                print(f"  Risk Level: {fhr_a.get('risk_level', 'N/A')}")
                
                if fhr_a.get('deviation', 0) != 0:
                    print(f"\n📈 DEVIATION:")
                    print(f"  Deviation: {fhr_a.get('deviation', 0):+.1f} bpm")
                    print(f"  Deviation %: {fhr_a.get('deviation_percentage', 0):+.2f}%")
                
                print(f"\n🏥 MEDICAL:")
                print(f"  Concern: {fhr_a.get('medical_concern', 'N/A')}")
                print(f"  Urgency: {fhr_a.get('urgency', 'N/A')}")
            
            print(f"\nGestation Period: {result['gestation_period']}")
            print("="*70)
            return True
        else:
            print(f"✗ Error response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("✗ Request timed out (30s). Server might be processing...")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*60)
    print("GarbhSuraksha API Test Suite")
    print("="*60)
    print()

    # Test 1: Health check
    health_ok = test_health_check()

    if not health_ok:
        print("\n❌ Health check failed. Cannot proceed with further tests.")
        print("\nMake sure the server is running:")
        print("  cd lib/backend")
        print("  python api_server.py")
        return

    # Test 2: Audio analysis
    print()
    analyze_ok = test_analyze_audio()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Audio Analysis: {'✓ PASS' if analyze_ok else '✗ FAIL'}")
    print("="*60)

    if health_ok and analyze_ok:
        print("\n🎉 All tests passed! The backend is working correctly.")
        print("\nYou can now:")
        print("  1. Run the Flutter app: flutter run")
        print("  2. Enter gestation period and record/upload audio")
        print("  3. Click 'Analyze' to get predictions")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()

