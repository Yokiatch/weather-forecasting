import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  CloudRain, Wind, Droplets, Cloud, MapPin,
  Loader2, Sparkles, Thermometer, Search, X,
  CheckCircle2, Circle, AlertTriangle
} from 'lucide-react';

const API_BASE = 'https://moosetape-weather-forecasting-api.hf.space';

const DEFAULT_CITY = {
  name: 'Greater Noida', country: 'India', admin1: 'Uttar Pradesh',
  lat: 28.4744, lon: 77.5040,
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function getDayName(offset) {
  if (offset === 1) return 'Tomorrow';
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return d.toLocaleDateString('en-US', { weekday: 'long' });
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function MetricCard({ icon, title, value }) {
  return (
    <div className="bg-slate-800/50 rounded-2xl p-4 md:p-6 border border-slate-700/50 flex flex-col md:flex-row items-start md:items-center gap-3 md:gap-4">
      <div className="p-3 bg-slate-700/50 rounded-xl text-blue-400 shrink-0">{icon}</div>
      <div>
        <p className="text-xs md:text-sm font-medium text-slate-400">{title}</p>
        <p className="text-lg md:text-xl font-semibold text-white">{value}</p>
      </div>
    </div>
  );
}

function Banner({ type, message }) {
  const s = {
    error:   'bg-red-900/40 border-red-700/50 text-red-300',
    warning: 'bg-yellow-900/40 border-yellow-700/50 text-yellow-300',
  };
  return (
    <div className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${s[type]}`}>
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

// Training progress stepper
const STEPS = [
  'Fetching historical weather data…',
  'Training XGBoost model…',
  'Fetching recent temperature history…',
  'Running 7-day forecast…',
];

function ProgressStepper({ currentStep, isDefaultCity }) {
  // For default city (Greater Noida) we skip steps 1-2
  const steps = isDefaultCity
    ? ['Loading pre-trained model…', 'Fetching recent temperature history…', 'Running 7-day forecast…']
    : STEPS;

  return (
    <div className="bg-slate-800/60 rounded-2xl border border-slate-700/50 p-6 space-y-4">
      <p className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
        Building forecast…
      </p>
      {steps.map((label, idx) => {
        const stepNum = idx + 1;
        const done    = currentStep > stepNum;
        const active  = currentStep === stepNum;
        return (
          <div key={idx} className="flex items-center gap-3">
            {done ? (
              <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
            ) : active ? (
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin shrink-0" />
            ) : (
              <Circle className="w-5 h-5 text-slate-600 shrink-0" />
            )}
            <span className={`text-sm ${done ? 'text-slate-500 line-through' : active ? 'text-white' : 'text-slate-600'}`}>
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// City search dropdown
function CitySearch({ onSelect }) {
  const [query, setQuery]       = useState('');
  const [results, setResults]   = useState([]);
  const [searching, setSearching] = useState(false);
  const [open, setOpen]         = useState(false);
  const debounceRef             = useRef(null);
  const wrapperRef              = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.trim().length < 2) { setResults([]); setOpen(false); return; }

    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const res = await axios.get(`${API_BASE}/api/geocode`, { params: { city: val } });
        setResults(res.data.results ?? []);
        setOpen(true);
      } catch {
        setResults([]);
      } finally {
        setSearching(false);
      }
    }, 400);
  };

  const handleSelect = (city) => {
    setQuery(`${city.name}, ${city.admin1 ? city.admin1 + ', ' : ''}${city.country}`);
    setOpen(false);
    setResults([]);
    onSelect(city);
  };

  const handleClear = () => {
    setQuery('');
    setResults([]);
    setOpen(false);
    onSelect(DEFAULT_CITY);
  };

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 focus-within:border-blue-500 transition-colors">
        {searching
          ? <Loader2 className="w-4 h-4 text-slate-400 animate-spin shrink-0" />
          : <Search className="w-4 h-4 text-slate-400 shrink-0" />
        }
        <input
          type="text"
          value={query}
          onChange={handleChange}
          placeholder="Search any city worldwide…"
          className="flex-1 bg-transparent text-white placeholder-slate-500 text-sm outline-none"
        />
        {query && (
          <button onClick={handleClear} className="text-slate-500 hover:text-white transition-colors">
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl z-50">
          {results.map((city, i) => (
            <button
              key={i}
              onClick={() => handleSelect(city)}
              className="w-full text-left px-4 py-3 hover:bg-slate-700 transition-colors flex items-start gap-3 border-b border-slate-700/50 last:border-0"
            >
              <MapPin className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">{city.name}</p>
                <p className="text-slate-400 text-xs">
                  {[city.admin1, city.country].filter(Boolean).join(', ')}
                </p>
              </div>
            </button>
          ))}
        </div>
      )}

      {open && results.length === 0 && !searching && query.length >= 2 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-400 text-sm shadow-2xl z-50">
          No cities found for "{query}"
        </div>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedCity, setSelectedCity] = useState(DEFAULT_CITY);

  const [currentWeather, setCurrentWeather] = useState(null);
  const [weatherError, setWeatherError]     = useState(null);
  const [loadingWeather, setLoadingWeather] = useState(true);

  const [forecast, setForecast]         = useState(null);
  const [forecastMae, setForecastMae]   = useState(null);
  const [predicting, setPredicting]     = useState(false);
  const [trainingStep, setTrainingStep] = useState(0);
  const [forecastError, setForecastError] = useState(null);

  const [expandedDay, setExpandedDay] = useState(null);

  const isDefaultCity = (c) =>
    Math.abs(c.lat - DEFAULT_CITY.lat) < 0.05 && Math.abs(c.lon - DEFAULT_CITY.lon) < 0.05;

  // Fetch live weather whenever city changes
  useEffect(() => {
    const fetchWeather = async () => {
      setLoadingWeather(true);
      setWeatherError(null);
      setCurrentWeather(null);
      setForecast(null);
      setForecastError(null);
      try {
        const res = await axios.get(`${API_BASE}/api/current_weather`, {
          params: { lat: selectedCity.lat, lon: selectedCity.lon },
        });
        setCurrentWeather(res.data.data);
      } catch (err) {
        setWeatherError(err?.response?.data?.detail ?? 'Could not fetch live weather.');
      } finally {
        setLoadingWeather(false);
      }
    };
    fetchWeather();
  }, [selectedCity]);

  // SSE-based train + predict
  const handlePredict = () => {
    if (!currentWeather) return;
    setPredicting(true);
    setForecast(null);
    setForecastError(null);
    setTrainingStep(1);
    setExpandedDay(null);

    const params = new URLSearchParams({
      lat:          selectedCity.lat,
      lon:          selectedCity.lon,
      current_temp: currentWeather.temperature,
      windspeed:    currentWeather.windspeed,
      humidity:     currentWeather.humidity,
      cloudcover:   currentWeather.cloud_cover,
    });

    const es = new EventSource(`${API_BASE}/api/train_and_predict?${params}`);

    es.addEventListener('step', (e) => {
      const data = JSON.parse(e.data);
      setTrainingStep(data.step);
    });

    es.addEventListener('result', (e) => {
      const data = JSON.parse(e.data);
      setForecast(data.forecast);
      setForecastMae(data.mae);
      setPredicting(false);
      setTrainingStep(0);
      es.close();
    });

    es.addEventListener('error', (e) => {
      try {
        const data = JSON.parse(e.data);
        setForecastError(data.message);
      } catch {
        setForecastError('Prediction failed. Please try again.');
      }
      setPredicting(false);
      setTrainingStep(0);
      es.close();
    });

    // Fallback if SSE connection itself errors
    es.onerror = () => {
      if (predicting) {
        setForecastError('Connection to server lost. Is the API running?');
        setPredicting(false);
        setTrainingStep(0);
        es.close();
      }
    };
  };

  const toggleDay = (d) => setExpandedDay(expandedDay === d ? null : d);

  const cityLabel = selectedCity.name +
    (selectedCity.admin1 ? `, ${selectedCity.admin1}` : '') +
    `, ${selectedCity.country}`;

  return (
    <div className="min-h-screen bg-slate-900 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-6">

        {/* ── Header ── */}
        <header className="pb-6 border-b border-slate-800">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 mb-5">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white mb-1">
                Weather Forecast
              </h1>
              <div className="flex items-center text-slate-400 gap-2 text-sm">
                <MapPin className="w-4 h-4 shrink-0" />
                <span>{cityLabel}</span>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={predicting || loadingWeather || !currentWeather}
              className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg shadow-blue-900/20 disabled:opacity-50 w-full md:w-auto"
            >
              {predicting
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Sparkles className="w-4 h-4" />}
              {predicting ? 'Building Forecast…' : 'Predict 7 Days'}
            </button>
          </div>

          {/* City search */}
          <CitySearch onSelect={(city) => { setSelectedCity(city); }} />
        </header>

        {/* ── Live Weather ── */}
        {loadingWeather ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        ) : weatherError ? (
          <Banner type="error" message={weatherError} />
        ) : currentWeather ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="col-span-1 bg-gradient-to-br from-slate-800 to-slate-800/50 rounded-3xl p-8 border border-slate-700 shadow-xl flex flex-col justify-center">
              <p className="text-slate-400 font-medium mb-4 uppercase tracking-wider text-sm">
                Current Temp
              </p>
              <div className="flex items-start gap-2">
                <span className="text-6xl font-black text-white tracking-tighter">
                  {currentWeather.temperature?.toFixed(1)}
                </span>
                <span className="text-2xl font-bold text-slate-500 mt-2">°C</span>
              </div>
            </div>

            <div className="col-span-1 md:col-span-2 grid grid-cols-2 gap-4 md:gap-6">
              <MetricCard icon={<Wind />}      title="Wind Speed"    value={`${currentWeather.windspeed} km/h`} />
              <MetricCard icon={<Droplets />}  title="Humidity"      value={`${currentWeather.humidity}%`} />
              <MetricCard icon={<Cloud />}     title="Cloud Cover"   value={`${currentWeather.cloud_cover}%`} />
              <MetricCard icon={<CloudRain />} title="Precipitation" value={`${currentWeather.precipitation} mm`} />
            </div>
          </div>
        ) : null}

        {/* ── Progress Stepper ── */}
        {predicting && (
          <ProgressStepper
            currentStep={trainingStep}
            isDefaultCity={isDefaultCity(selectedCity)}
          />
        )}

        {/* ── Errors ── */}
        {forecastError && <Banner type="error" message={forecastError} />}

        {/* ── 7-Day Forecast ── */}
        {forecast && (
          <div className="bg-slate-800/40 rounded-3xl p-6 md:p-8 border border-slate-700/50 shadow-xl">
            <div className="flex items-start justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">7-Day Forecast</h2>
              {forecastMae && (
                <span className="text-xs text-slate-500 bg-slate-700/50 px-3 py-1 rounded-full">
                  Model MAE: ±{forecastMae}°C
                </span>
              )}
            </div>

            <div className="flex flex-col gap-3">
              {forecast.map((day) => (
                <div key={day.day} className="bg-slate-800/60 rounded-2xl border border-slate-700/50 overflow-hidden">
                  <div
                    onClick={() => toggleDay(day.day)}
                    className="flex items-center justify-between p-4 cursor-pointer hover:bg-slate-700/50 transition-colors select-none"
                  >
                    <div className="w-1/3 md:w-1/4 text-white font-medium text-lg">
                      {getDayName(day.day)}
                    </div>
                    <div className="w-1/3 flex items-center justify-center gap-3">
                      <Thermometer className="w-5 h-5 text-blue-400" />
                      <span className="text-white font-bold text-lg">{day.temp}°</span>
                    </div>
                    <div className="w-1/3 flex justify-end items-center gap-1 md:gap-3 text-slate-400 font-medium">
                      <span>L: {day.low}°</span>
                      <div className="hidden md:block w-24 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-500 to-orange-400 w-full opacity-70" />
                      </div>
                      <span className="text-white">H: {day.high}°</span>
                    </div>
                  </div>

                  {expandedDay === day.day && (
                    <div className="p-4 bg-slate-800/80 border-t border-slate-700/50 grid grid-cols-3 gap-4">
                      {[
                        { icon: <Wind className="w-5 h-5 text-slate-400 mb-1" />, label: 'Wind',   val: day.wind  != null ? `${day.wind} km/h` : '--' },
                        { icon: <Cloud className="w-5 h-5 text-slate-400 mb-1" />, label: 'Clouds', val: day.clouds != null ? `${day.clouds}%` : '--' },
                        { icon: <CloudRain className="w-5 h-5 text-slate-400 mb-1" />, label: 'Precip', val: day.precip != null ? `${day.precip} mm` : '--' },
                      ].map(({ icon, label, val }) => (
                        <div key={label} className="flex flex-col items-center p-3 bg-slate-700/30 rounded-xl">
                          {icon}
                          <span className="text-xs text-slate-500 uppercase font-semibold">{label}</span>
                          <span className="text-white font-medium">{val}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <p className="text-xs text-slate-500 mt-4">
              Temperature predicted by XGBoost trained on 4 years of historical data for {selectedCity.name}.
              Wind, clouds &amp; precipitation from Open-Meteo.
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
