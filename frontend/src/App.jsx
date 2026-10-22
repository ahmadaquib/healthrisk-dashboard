import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Database,
  FileText,
  RotateCcw,
  Shield,
  ShieldAlert,
  Zap
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine
} from 'recharts';

export default function App() {
  const [diseases, setDiseases] = useState([]);
  const [selectedDiseaseId, setSelectedDiseaseId] = useState('diabetes');
  const [activeTab, setActiveTab] = useState('risk'); // 'risk' | 'performance'
  const [formValues, setFormValues] = useState({});
  const [fieldErrors, setFieldErrors] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isDataOpen, setIsDataOpen] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  // Fetch disease list on mount
  useEffect(() => {
    fetchDiseases();
  }, []);

  // Fetch metrics and reset form when selected disease changes
  useEffect(() => {
    if (!selectedDiseaseId || diseases.length === 0) return;
    const disease = diseases.find((d) => d.id === selectedDiseaseId);
    if (disease) {
      const initial = {};
      disease.fields.forEach((f) => {
        initial[f.name] = f.default;
      });
      setFormValues(initial);
      setFieldErrors({});
      setPrediction(null);
      setError(null);
      fetchMetrics(selectedDiseaseId);
    }
  }, [selectedDiseaseId, diseases]);

  const fetchDiseases = async () => {
    try {
      setError(null);
      const res = await fetch('/api/diseases');
      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();
      setDiseases(data);
      if (data.length > 0) {
        setSelectedDiseaseId(data[0].id);
      }
    } catch (err) {
      setError(`Failed to connect to backend: ${err.message}. Please ensure the FastAPI server is running on port 8000.`);
    }
  };

  const fetchMetrics = async (diseaseId) => {
    try {
      const res = await fetch(`/api/metrics/${diseaseId}`);
      if (!res.ok) throw new Error(`Failed to load metrics (${res.status})`);
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      setError(`Failed to load model metrics: ${err.message}`);
    }
  };

  const validateField = (field, value) => {
    if (value === '' || value === null || value === undefined || isNaN(value)) {
      return 'Value is required';
    }
    const num = Number(value);
    if (field.min_sane !== undefined && num < field.min_sane) {
      return `Must be >= ${field.min_sane} ${field.unit || ''}`.trim();
    }
    if (field.max_sane !== undefined && num > field.max_sane) {
      return `Must be <= ${field.max_sane} ${field.unit || ''}`.trim();
    }
    return null;
  };

  const handleInputChange = (field, value) => {
    const parsed = value === '' ? '' : Number(value);
    setFormValues((prev) => ({
      ...prev,
      [field.name]: parsed,
    }));

    const validationMsg = validateField(field, parsed);
    setFieldErrors((prev) => ({
      ...prev,
      [field.name]: validationMsg,
    }));
  };

  const loadExample = (type) => {
    const currentDisease = diseases.find((d) => d.id === selectedDiseaseId);
    if (!currentDisease?.example_patients) return;
    const example = currentDisease.example_patients[type];
    if (example?.values) {
      setFormValues({ ...example.values });
      setFieldErrors({});
      setPrediction(null);
      setError(null);
    }
  };

  const handleReset = () => {
    const currentDisease = diseases.find((d) => d.id === selectedDiseaseId);
    if (!currentDisease) return;
    const initial = {};
    currentDisease.fields.forEach((f) => {
      initial[f.name] = f.default;
    });
    setFormValues(initial);
    setFieldErrors({});
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    const currentDisease = diseases.find((d) => d.id === selectedDiseaseId);
    if (!currentDisease) return;

    // Validate all fields before submission
    const errors = {};
    let hasError = false;
    currentDisease.fields.forEach((f) => {
      const err = validateField(f, formValues[f.name]);
      if (err) {
        errors[f.name] = err;
        hasError = true;
      }
    });

    setFieldErrors(errors);
    if (hasError) {
      setError('Please correct the highlighted input fields within physiologically plausible bounds before running risk assessment.');
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      const res = await fetch(`/api/predict/${selectedDiseaseId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formValues),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  const currentDisease = diseases.find((d) => d.id === selectedDiseaseId);

  // Probability display formatter: >99% if >99.9%, <1% if <0.1%
  const formatProbability = (prob) => {
    if (prob >= 0.999) return '>99%';
    if (prob <= 0.001) return '<1%';
    return `${(prob * 100).toFixed(1)}%`;
  };

  // Plain-language clinical interpretation
  const getPlainLanguageSentence = (riskLevel) => {
    if (riskLevel === 'High') {
      return 'The model estimates a high likelihood based on the entered values.';
    }
    if (riskLevel === 'Moderate') {
      return 'The model estimates a moderate likelihood based on the entered values.';
    }
    return 'The model estimates a low likelihood based on the entered values.';
  };

  // Recharts Custom Tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div style={{
          background: '#0f172a',
          color: '#ffffff',
          padding: '8px 12px',
          borderRadius: '8px',
          fontSize: '12px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}>
          <p style={{ fontWeight: 700, marginBottom: '4px' }}>{item.label}</p>
          <p>Entered Value: <strong>{item.value} {item.unit}</strong></p>
          <p>Direction: <strong style={{ color: item.contribution > 0 ? '#f87171' : '#4ade80' }}>{item.direction}</strong></p>
          <p>Impact Score: <strong>{item.impact}</strong></p>
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      {/* Sticky Top Bar */}
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-badge">
              <Activity size={22} />
            </div>
            <div className="brand-text">
              <h1>AI Disease Risk Prediction Dashboard</h1>
              <p>Clinical Decision Support Dashboard</p>
            </div>
          </div>
          <div className="demo-pill">
            <Shield size={13} />
            <span>Educational demo only. Not a medical diagnosis.</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-wrapper">
        {/* Navigation Tabs */}
        <div className="tab-bar">
          <button
            id="tab-risk-assessment"
            className={`tab-button ${activeTab === 'risk' ? 'active' : ''}`}
            onClick={() => setActiveTab('risk')}
          >
            <Activity size={17} />
            <span>Patient Risk Assessment</span>
          </button>
          <button
            id="tab-model-performance"
            className={`tab-button ${activeTab === 'performance' ? 'active' : ''}`}
            onClick={() => setActiveTab('performance')}
          >
            <BarChart3 size={17} />
            <span>Model Performance &amp; Evaluation</span>
          </button>
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="error-banner">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Disease Selector: Two Large Cards with Active States */}
        <section className="disease-cards-grid">
          {diseases.map((d) => {
            const isSelected = d.id === selectedDiseaseId;
            return (
              <div
                key={d.id}
                id={`disease-card-${d.id}`}
                className={`disease-select-card ${isSelected ? 'active' : ''}`}
                onClick={() => setSelectedDiseaseId(d.id)}
              >
                <div className="disease-card-icon">
                  <Activity size={24} />
                </div>
                <div className="disease-card-content">
                  <div className="disease-card-title-row">
                    <h2>{d.name}</h2>
                    {isSelected && <span className="active-indicator">Selected</span>}
                  </div>
                  <p className="disease-card-desc">{d.description}</p>
                </div>
              </div>
            );
          })}
        </section>

        {activeTab === 'risk' ? (
          /* TAB 1: Patient Form & Assessment Result */
          <>
            <div className="content-grid">
              {/* Form Card */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">
                    <FileText size={18} color="#0d9488" />
                    <span>Patient Health Parameters</span>
                  </h3>
                </div>

                {/* Example Presets & Reset Row */}
                <div className="action-presets-row">
                  <button
                    type="button"
                    id="btn-load-high-risk"
                    className="btn-secondary high"
                    onClick={() => loadExample('high_risk')}
                  >
                    <Zap size={14} />
                    <span>Load high-risk example</span>
                  </button>
                  <button
                    type="button"
                    id="btn-load-low-risk"
                    className="btn-secondary low"
                    onClick={() => loadExample('low_risk')}
                  >
                    <CheckCircle size={14} />
                    <span>Load low-risk example</span>
                  </button>
                  <button
                    type="button"
                    id="btn-reset-form"
                    className="btn-secondary"
                    onClick={handleReset}
                  >
                    <RotateCcw size={14} />
                    <span>Reset</span>
                  </button>
                </div>

                {/* 2-Column Dynamic Form */}
                {currentDisease ? (
                  <form onSubmit={handlePredict}>
                    <div className="form-grid-2col">
                      {currentDisease.fields.map((field) => {
                        const hasErr = Boolean(fieldErrors[field.name]);
                        return (
                          <div key={field.name} className="input-group">
                            <div className="input-label-row">
                              <label htmlFor={`input-${field.name}`} className="input-label">
                                {field.label}
                              </label>
                              {field.unit && <span className="input-unit">{field.unit}</span>}
                            </div>
                            <input
                              id={`input-${field.name}`}
                              type="number"
                              step={field.step || 'any'}
                              value={formValues[field.name] ?? ''}
                              onChange={(e) => handleInputChange(field, e.target.value)}
                              className={`input-control ${hasErr ? 'has-error' : ''}`}
                              required
                            />
                            {hasErr ? (
                              <span className="inline-error">{fieldErrors[field.name]}</span>
                            ) : (
                              <span className="input-hint">
                                Range: {field.training_min} - {field.training_max}
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    <button
                      type="submit"
                      id="btn-run-prediction"
                      className="btn-primary"
                      disabled={predicting}
                    >
                      {predicting ? (
                        <>
                          <div className="spinner" />
                          <span>Evaluating Risk...</span>
                        </>
                      ) : (
                        <>
                          <Activity size={18} />
                          <span>Predict risk</span>
                        </>
                      )}
                    </button>
                  </form>
                ) : (
                  <p>Loading disease parameters...</p>
                )}
              </div>

              {/* Result Card */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">
                    <Shield size={18} color="#0d9488" />
                    <span>Risk Result &amp; Explanation</span>
                  </h3>
                </div>

                {prediction ? (
                  <div className="result-container">
                    {/* Risk Badge & Formatted Probability */}
                    <div className={`result-header-badge-card ${prediction.risk_level.toLowerCase()}`}>
                      <div>
                        <span className={`risk-badge ${prediction.risk_level.toLowerCase()}`}>
                          {prediction.risk_level} Risk
                        </span>
                        <p style={{ fontSize: '0.8125rem', color: '#475569', marginTop: '6px' }}>
                          Evaluated Disease Probability
                        </p>
                      </div>
                      <div className="risk-number">
                        {formatProbability(prediction.probability)}
                      </div>
                    </div>

                    {/* Progress Gauge */}
                    <div>
                      <div className="gauge-track">
                        <div
                          className={`gauge-fill ${prediction.risk_level.toLowerCase()}`}
                          style={{
                            width: `${Math.min(Math.max(prediction.probability * 100, 2), 100)}%`
                          }}
                        />
                      </div>
                      <div className="gauge-labels">
                        <span>Low (&lt;33%)</span>
                        <span>Moderate (33-66%)</span>
                        <span>High (&gt;66%)</span>
                      </div>
                    </div>

                    {/* Plain Language Sentence */}
                    <div className="plain-language-card">
                      {getPlainLanguageSentence(prediction.risk_level)}
                    </div>

                    {/* Warnings Box in Amber */}
                    {prediction.warnings && prediction.warnings.length > 0 && (
                      <div className="warnings-box">
                        <div className="warnings-header">
                          <AlertTriangle size={15} />
                          <span>Uncertainty Notice ({prediction.warnings.length})</span>
                        </div>
                        <div className="warnings-items">
                          {prediction.warnings.map((w, idx) => (
                            <span key={idx}>&bull; {w}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recharts Factors Bar Chart */}
                    <div className="factors-box">
                      <div className="factors-header">
                        <span>Top 5 Contributing Factors</span>
                        <div className="factors-legend">
                          <div className="legend-badge">
                            <span className="dot red" />
                            <span>Increases risk</span>
                          </div>
                          <div className="legend-badge">
                            <span className="dot green" />
                            <span>Decreases risk</span>
                          </div>
                        </div>
                      </div>

                      <div style={{ width: '100%', height: 210 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={prediction.top_factors}
                            margin={{ top: 5, right: 30, left: 35, bottom: 5 }}
                          >
                            <XAxis
                              type="number"
                              tick={{ fontSize: 11, fill: '#64748b' }}
                            />
                            <YAxis
                              type="category"
                              dataKey="label"
                              tick={{ fontSize: 11, fill: '#334155' }}
                              width={110}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine x={0} stroke="#cbd5e1" strokeDasharray="3 3" />
                            <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                              {prediction.top_factors.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.contribution > 0 ? '#ef4444' : '#10b981'}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="empty-state">
                    <ShieldAlert className="empty-state-icon" />
                    <h3>No Assessment Generated Yet</h3>
                    <p>Enter patient health parameters or click a preset above, then click <strong>"Predict risk"</strong>.</p>
                  </div>
                )}
              </div>
            </div>

            {/* About the Dataset Section (Collapsible Card under the Form) */}
            <div className="collapsible-card">
              <button
                type="button"
                className="collapsible-header"
                onClick={() => setIsDataOpen(!isDataOpen)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Database size={18} color="#0d9488" />
                  <span>About the Dataset: {currentDisease?.name}</span>
                </div>
                {isDataOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>

              {isDataOpen && metrics && (
                <div className="collapsible-body">
                  <div className="dataset-facts-grid">
                    <div className="fact-item">
                      <div className="fact-title">Dataset Name</div>
                      <div className="fact-value">{metrics.dataset_info?.name || currentDisease?.name}</div>
                    </div>
                    <div className="fact-item">
                      <div className="fact-title">Total Records</div>
                      <div className="fact-value">{metrics.dataset_info?.total_records || (metrics.train_size + metrics.test_size)} records</div>
                    </div>
                    <div className="fact-item">
                      <div className="fact-title">Number of Features</div>
                      <div className="fact-value">{metrics.features?.length || 0} features</div>
                    </div>
                    <div className="fact-item">
                      <div className="fact-title">Class Balance</div>
                      <div className="fact-value">
                        {metrics.dataset_info?.class_balance
                          ? Object.entries(metrics.dataset_info.class_balance).map(([k, v]) => `${k}: ${v}`).join(' | ')
                          : `${metrics.train_size} train / ${metrics.test_size} test`}
                      </div>
                    </div>
                  </div>

                  <p>
                    {selectedDiseaseId === 'diabetes' ? (
                      <>
                        <strong>Diabetes:</strong> Pima Indians Diabetes Dataset (originally from the National Institute of Diabetes and Digestive and Kidney Diseases), loaded via OpenML &quot;diabetes&quot; v1; 768 records, 8 features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age); target = diabetes diagnosis (1 = diabetic); zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing and median-imputed.
                      </>
                    ) : (
                      <>
                        <strong>Breast Cancer:</strong> Breast Cancer Wisconsin (Diagnostic) dataset, loaded from scikit-learn (sklearn.datasets.load_breast_cancer), originally from the UCI Machine Learning Repository; 569 records, 10 &quot;mean&quot; features used (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension); target = 1 means malignant.
                      </>
                    )}
                  </p>
                </div>
              )}
            </div>
          </>
        ) : (
          /* TAB 2: Model Performance & Evaluation */
          <div className="perf-grid">
            {metrics ? (
              <>
                {/* Metric Cards Row */}
                <div className="metric-cards-row">
                  <div className="metric-card">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-number">{(metrics.holdout_metrics.accuracy * 100).toFixed(2)}%</span>
                    <span className="metric-desc">Overall correctness</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Precision</span>
                    <span className="metric-number">{(metrics.holdout_metrics.precision * 100).toFixed(2)}%</span>
                    <span className="metric-desc">Positive predictive value</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Recall</span>
                    <span className="metric-number">{(metrics.holdout_metrics.recall * 100).toFixed(2)}%</span>
                    <span className="metric-desc">True positive rate (Sensitivity)</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">F1-Score</span>
                    <span className="metric-number">{metrics.holdout_metrics.f1.toFixed(4)}</span>
                    <span className="metric-desc">Harmonic mean balance</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">ROC-AUC</span>
                    <span className="metric-number">{metrics.holdout_metrics.roc_auc.toFixed(4)}</span>
                    <span className="metric-desc">Discriminatory capacity</span>
                  </div>
                </div>

                {/* Comparison Benchmark Table */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <BarChart3 size={18} color="#0d9488" />
                      <span>Cross-Validation Benchmark (RepeatedStratifiedKFold: 5 folds &times; 10 repeats = 50 runs)</span>
                    </h3>
                  </div>

                  <div className="table-wrap">
                    <table className="comparison-table">
                      <thead>
                        <tr>
                          <th>Model Candidate</th>
                          <th>Mean Accuracy (&plusmn; Std)</th>
                          <th>Mean F1-Score (&plusmn; Std)</th>
                          <th>Mean ROC-AUC (&plusmn; Std)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(metrics.cv_comparison).map(([mName, vals]) => {
                          const isLR = mName.includes('Logistic');
                          return (
                            <tr key={mName} className={isLR ? 'highlight-row' : ''}>
                              <td>
                                {mName}
                                {isLR && (
                                  <span style={{ marginLeft: '8px', fontSize: '0.72rem', background: '#0d9488', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>
                                    Production Model
                                  </span>
                                )}
                              </td>
                              <td className={isLR ? 'best-val' : ''}>
                                {(vals.accuracy_mean * 100).toFixed(2)}% (&plusmn; {(vals.accuracy_std * 100).toFixed(2)}%)
                              </td>
                              <td className={isLR ? 'best-val' : ''}>
                                {vals.f1_mean.toFixed(3)} (&plusmn; {vals.f1_std.toFixed(3)})
                              </td>
                              <td className={isLR ? 'best-val' : ''}>
                                {vals.roc_auc_mean.toFixed(3)} (&plusmn; {vals.roc_auc_std.toFixed(3)})
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* 2x2 Confusion Matrix */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <Shield size={18} color="#0d9488" />
                      <span>Holdout Test Set Confusion Matrix (2&times;2 Grid)</span>
                    </h3>
                  </div>

                  <div className="matrix-container">
                    <div className="matrix-cell tn">
                      <span className="matrix-label">True Negatives (TN)</span>
                      <div className="matrix-count">{metrics.holdout_metrics.confusion_matrix[0][0]}</div>
                      <span className="matrix-info">Correctly identified non-disease</span>
                    </div>

                    <div className="matrix-cell error">
                      <span className="matrix-label" style={{ color: '#be123c' }}>False Positives (FP)</span>
                      <div className="matrix-count" style={{ color: '#be123c' }}>{metrics.holdout_metrics.confusion_matrix[0][1]}</div>
                      <span className="matrix-info">False alarm (Type I error)</span>
                    </div>

                    <div className="matrix-cell error">
                      <span className="matrix-label" style={{ color: '#be123c' }}>False Negatives (FN)</span>
                      <div className="matrix-count" style={{ color: '#be123c' }}>{metrics.holdout_metrics.confusion_matrix[1][0]}</div>
                      <span className="matrix-info">Missed diagnosis (Type II error)</span>
                    </div>

                    <div className="matrix-cell tp">
                      <span className="matrix-label" style={{ color: '#047857' }}>True Positives (TP)</span>
                      <div className="matrix-count" style={{ color: '#047857' }}>{metrics.holdout_metrics.confusion_matrix[1][1]}</div>
                      <span className="matrix-info">Correctly identified disease</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <p>Loading model performance metrics...</p>
            )}
          </div>
        )}

        {/* Footer */}
        <footer className="app-footer">
          <p>Educational demo only. Not a medical diagnosis.</p>
        </footer>
      </main>
    </div>
  );
}
