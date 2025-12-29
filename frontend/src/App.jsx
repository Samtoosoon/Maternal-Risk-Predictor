import { useState, useEffect } from 'react'
import './App.css'

const API_BASE = 'http://localhost:5000/api'

// Risk Score Display Component
function RiskScoreDisplay({ score, category }) {
  const getScoreColor = () => {
    if (category === 'High') return 'var(--risk-high)'
    if (category === 'Medium') return 'var(--risk-medium)'
    return 'var(--risk-low)'
  }

  const getGradient = () => {
    if (category === 'High') return 'linear-gradient(135deg, #ff4757, #ff6b7a)'
    if (category === 'Medium') return 'linear-gradient(135deg, #ffa502, #ffbe00)'
    return 'linear-gradient(135deg, #2ed573, #7bed9f)'
  }

  return (
    <div className="risk-score-container">
      <div
        className="risk-score-circle"
        style={{ background: getGradient() }}
      >
        <div className="risk-score-inner">
          <span className="risk-score-value">{score}</span>
          <span className="risk-score-max">/100</span>
        </div>
      </div>
      <div
        className="risk-category-badge"
        style={{ backgroundColor: getScoreColor() }}
      >
        {category} Risk
      </div>
    </div>
  )
}

// Risk Factors Component
function RiskFactors({ factors }) {
  const getSeverityIcon = (severity) => {
    if (severity === 'high') return '🔴'
    if (severity === 'medium') return '🟡'
    return '🟢'
  }

  if (!factors || factors.length === 0) {
    return (
      <div className="risk-factors-empty">
        <span className="success-icon">✓</span>
        <p>No significant risk factors identified</p>
      </div>
    )
  }

  return (
    <div className="risk-factors">
      <h3>Contributing Factors</h3>
      <div className="factors-list">
        {factors.map((factor, idx) => (
          <div key={idx} className={`factor-item factor-${factor.severity}`}>
            <span className="factor-icon">{getSeverityIcon(factor.severity)}</span>
            <div className="factor-content">
              <span className="factor-name">{factor.factor}</span>
              <span className="factor-description">{factor.description}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Recommendations Component
function Recommendations({ recommendations }) {
  const getPriorityStyles = (priority) => {
    if (priority === 'urgent') return { bg: '#ffe8e8', border: '#ff4757' }
    if (priority === 'high') return { bg: '#fff3e0', border: '#ffa502' }
    if (priority === 'important') return { bg: '#e3f2fd', border: '#2196f3' }
    return { bg: '#f5f5f5', border: '#9e9e9e' }
  }

  if (!recommendations || recommendations.length === 0) return null

  return (
    <div className="recommendations">
      <h3>Clinical Recommendations</h3>
      <div className="recommendations-list">
        {recommendations.map((rec, idx) => {
          const styles = getPriorityStyles(rec.priority)
          return (
            <div
              key={idx}
              className="recommendation-item"
              style={{
                backgroundColor: styles.bg,
                borderLeftColor: styles.border
              }}
            >
              <span className="recommendation-priority">{rec.priority.toUpperCase()}</span>
              <span className="recommendation-action">{rec.action}</span>
              <span className="recommendation-description">{rec.description}</span>
              {rec.source && (
                <span className="recommendation-source">Source: {rec.source}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Sample Profiles Selector
function SampleProfiles({ profiles, onSelect }) {
  return (
    <div className="sample-profiles">
      <h3>Demo Profiles</h3>
      <p className="profiles-description">Click to auto-fill with sample data</p>
      <div className="profiles-grid">
        {profiles.map((profile, idx) => (
          <button
            key={idx}
            className="profile-card"
            onClick={() => onSelect(profile.data)}
          >
            <span className="profile-name">{profile.name}</span>
            <span className="profile-desc">{profile.description}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

// Main App Component
function App() {
  const [formData, setFormData] = useState({
    age: 25,
    systolic_blood_pressure: 120,
    diastolic_blood_pressure: 80,
    blood_sugar: 85,
    hemoglobin: 12,
    heart_rate: 75,
    bmi: 22.5,
    parity: 0,
    anc_visits: 4,
    location: 1
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sampleProfiles, setSampleProfiles] = useState([])
  const [featureInfo, setFeatureInfo] = useState({})

  useEffect(() => {
    // Fetch sample profiles and feature info on mount
    fetch(`${API_BASE}/sample-profiles`)
      .then(res => res.json())
      .then(data => setSampleProfiles(data))
      .catch(err => console.error('Failed to load profiles:', err))

    fetch(`${API_BASE}/feature-info`)
      .then(res => res.json())
      .then(data => setFeatureInfo(data))
      .catch(err => console.error('Failed to load feature info:', err))
  }, [])

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
    setPrediction(null) // Clear previous prediction when form changes
  }

  const handleProfileSelect = (profileData) => {
    setFormData(profileData)
    setPrediction(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.error || 'Prediction failed')
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getInputProps = (field) => {
    const info = featureInfo[field] || {}
    return {
      min: info.min,
      max: info.max,
      step: field === 'bmi' || field === 'hemoglobin' ? 0.1 : 1
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-text">
              <h1>Maternal Health Risk Predictor</h1>
              <p>AI-powered pregnancy risk assessment for healthcare professionals</p>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="content-grid">
          {/* Left Panel - Input Form */}
          <section className="form-section">
            <div className="section-header">
              <h2>Patient Information</h2>
              <p>Enter maternal health indicators for risk assessment</p>
            </div>

            {sampleProfiles.length > 0 && (
              <SampleProfiles
                profiles={sampleProfiles}
                onSelect={handleProfileSelect}
              />
            )}

            <form onSubmit={handleSubmit} className="patient-form">
              <div className="form-group">
                <label htmlFor="age">
                  Age (years)
                  <span className="normal-range">Normal: 18-35</span>
                </label>
                <input
                  type="number"
                  id="age"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', parseFloat(e.target.value))}
                  {...getInputProps('age')}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="systolic_bp">
                    Systolic BP (mmHg)
                    <span className="normal-range">Normal: 90-120</span>
                  </label>
                  <input
                    type="number"
                    id="systolic_bp"
                    value={formData.systolic_blood_pressure}
                    onChange={(e) => handleInputChange('systolic_blood_pressure', parseFloat(e.target.value))}
                    {...getInputProps('systolic_blood_pressure')}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="diastolic_bp">
                    Diastolic BP (mmHg)
                    <span className="normal-range">Normal: 60-80</span>
                  </label>
                  <input
                    type="number"
                    id="diastolic_bp"
                    value={formData.diastolic_blood_pressure}
                    onChange={(e) => handleInputChange('diastolic_blood_pressure', parseFloat(e.target.value))}
                    {...getInputProps('diastolic_blood_pressure')}
                    required
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="blood_sugar">
                    Blood Sugar (mg/dL)
                    <span className="normal-range">Normal: 70-100</span>
                  </label>
                  <input
                    type="number"
                    id="blood_sugar"
                    value={formData.blood_sugar}
                    onChange={(e) => handleInputChange('blood_sugar', parseFloat(e.target.value))}
                    {...getInputProps('blood_sugar')}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="hemoglobin">
                    Hemoglobin (g/dL)
                    <span className="normal-range">Normal: 11-14</span>
                  </label>
                  <input
                    type="number"
                    id="hemoglobin"
                    value={formData.hemoglobin}
                    onChange={(e) => handleInputChange('hemoglobin', parseFloat(e.target.value))}
                    {...getInputProps('hemoglobin')}
                    step="0.1"
                    required
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="heart_rate">
                    Heart Rate (bpm)
                    <span className="normal-range">Normal: 60-100</span>
                  </label>
                  <input
                    type="number"
                    id="heart_rate"
                    value={formData.heart_rate}
                    onChange={(e) => handleInputChange('heart_rate', parseFloat(e.target.value))}
                    {...getInputProps('heart_rate')}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="bmi">
                    BMI
                    <span className="normal-range">Normal: 18.5-24.9</span>
                  </label>
                  <input
                    type="number"
                    id="bmi"
                    value={formData.bmi}
                    onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value))}
                    {...getInputProps('bmi')}
                    step="0.1"
                    required
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="parity">
                    Previous Pregnancies
                    <span className="normal-range">Normal: 0-4</span>
                  </label>
                  <input
                    type="number"
                    id="parity"
                    value={formData.parity}
                    onChange={(e) => handleInputChange('parity', parseInt(e.target.value))}
                    {...getInputProps('parity')}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="anc_visits">
                    ANC Visits
                    <span className="normal-range">Recommended: 4+</span>
                  </label>
                  <input
                    type="number"
                    id="anc_visits"
                    value={formData.anc_visits}
                    onChange={(e) => handleInputChange('anc_visits', parseInt(e.target.value))}
                    {...getInputProps('anc_visits')}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="location">Location</label>
                <select
                  id="location"
                  value={formData.location}
                  onChange={(e) => handleInputChange('location', parseInt(e.target.value))}
                >
                  <option value={0}>Rural</option>
                  <option value={1}>Urban</option>
                </select>
              </div>

              <button
                type="submit"
                className="submit-btn"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🔬</span>
                    Assess Risk
                  </>
                )}
              </button>
            </form>
          </section>

          {/* Right Panel - Results */}
          <section className="results-section">
            <div className="section-header">
              <h2>Risk Assessment</h2>
              <p>AI-powered prediction based on medical indicators</p>
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <p>{error}</p>
              </div>
            )}

            {!prediction && !error && (
              <div className="placeholder-result">
                <div className="placeholder-icon">📊</div>
                <h3>Enter Patient Data</h3>
                <p>Fill in the maternal health indicators on the left and click "Assess Risk" to get an AI-powered risk prediction.</p>
                <div className="placeholder-features">
                  <div className="feature-item">
                    <span className="feature-icon">🎯</span>
                    <span>Risk Score 0-100</span>
                  </div>
                  <div className="feature-item">
                    <span className="feature-icon">📋</span>
                    <span>Contributing Factors</span>
                  </div>
                  <div className="feature-item">
                    <span className="feature-icon">💡</span>
                    <span>Recommendations</span>
                  </div>
                </div>
              </div>
            )}

            {prediction && (
              <div className="prediction-results">
                <RiskScoreDisplay
                  score={prediction.risk_score}
                  category={prediction.risk_category}
                />

                <RiskFactors factors={prediction.risk_factors} />

                <Recommendations recommendations={prediction.recommendations} />
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        <p>
          <strong>Disclaimer:</strong> This tool is for screening purposes only and does not replace professional medical advice.
          All high-risk cases should be referred to qualified healthcare professionals.
        </p>
        <p className="footer-credits">
          Built with ❤️ for maternal health | Powered by Machine Learning
        </p>
      </footer>
    </div>
  )
}

export default App
