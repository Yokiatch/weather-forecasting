import { useState, useEffect } from 'react';
import axios from 'axios';
import { CloudRain, Wind, Droplets, Cloud, MapPin, Loader2, Sparkles, Sun, Thermometer } from 'lucide-react';

export default function App() {
  const [currentWeather, setCurrentWeather] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  
  // NEW: State to track which day is expanded
  const [expandedDay, setExpandedDay] = useState(null);

  // Fetch live weather on load
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const response = await axios.get('https://weather-forecasting-5o4n.onrender.com/api/current_weather');
        setCurrentWeather(response.data.data);
      } catch (error) {
        console.error("Failed to fetch weather:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchWeather();
  }, []);

  // Trigger XGBoost prediction
  const handlePredict = async () => {
    if (!currentWeather) return;
    setPredicting(true);
    try {
      const payload = {
        current_temp: currentWeather.temperature,
        windspeed: currentWeather.windspeed,
        humidity: currentWeather.humidity,
        cloudcover: currentWeather.cloud_cover
      };
      const response = await axios.post('https://weather-forecasting-5o4n.onrender.com/api/predict_7_days', payload);
      
      setForecast(response.data.forecast);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setPredicting(false);
    }
  };

  // Helper function to get the actual name of the day
  const getDayName = (offset) => {
    const date = new Date();
    date.setDate(date.getDate() + offset);
    if (offset === 1) return "Tomorrow";
    return date.toLocaleDateString('en-US', { weekday: 'long' });
  };

  // NEW: Helper function to toggle the expanded row
  const toggleDay = (dayNum) => {
    setExpandedDay(expandedDay === dayNum ? null : dayNum);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900">
        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 pb-6 border-b border-slate-800">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Weather Forecast</h1>
            <div className="flex items-center text-slate-400 gap-2">
              <MapPin className="w-4 h-4" />
              <span>Greater Noida, India</span>
            </div>
          </div>
          <button 
            onClick={handlePredict}
            disabled={predicting}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg shadow-blue-900/20 disabled:opacity-50 w-full md:w-auto justify-center"
          >
            {predicting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {predicting ? 'Running Model...' : 'Predict 7 Days'}
          </button>
        </header>

        {/* Current Weather Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="col-span-1 bg-linear-to-br from-slate-800 to-slate-800/50 rounded-3xl p-8 border border-slate-700 shadow-xl flex flex-col justify-center">
            <p className="text-slate-400 font-medium mb-4 uppercase tracking-wider text-sm">Current Temp</p>
            <div className="flex items-start gap-2">
              <span className="text-6xl font-black text-white tracking-tighter">
                {currentWeather?.temperature?.toFixed(1)}
              </span>
              <span className="text-2xl font-bold text-slate-500 mt-2">°C</span>
            </div>
          </div>

          <div className="col-span-1 md:col-span-2 grid grid-cols-2 gap-4 md:gap-6">
            <MetricCard icon={<Wind />} title="Wind Speed" value={`${currentWeather?.windspeed} km/h`} />
            <MetricCard icon={<Droplets />} title="Humidity" value={`${currentWeather?.humidity}%`} />
            <MetricCard icon={<Cloud />} title="Cloud Cover" value={`${currentWeather?.cloud_cover}%`} />
            <MetricCard icon={<CloudRain />} title="Precipitation" value={`${currentWeather?.precipitation} mm`} />
          </div>
        </div>

        {/* UPDATED 7-Day Forecast List with Expanding Rows */}
        {forecast && (
          <div className="bg-slate-800/40 rounded-3xl p-6 md:p-8 border border-slate-700/50 shadow-xl mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <h2 className="text-xl font-semibold text-white mb-6">7-Day Forecast</h2>
            
            <div className="flex flex-col gap-3">
              {forecast.map((day) => (
                <div 
                  key={day.day} 
                  className="bg-slate-800/60 transition-colors rounded-2xl border border-slate-700/50 overflow-hidden"
                >
                  {/* Clickable Header Row */}
                  <div 
                    onClick={() => toggleDay(day.day)}
                    className="flex items-center justify-between p-4 cursor-pointer hover:bg-slate-700/50"
                  >
                    {/* Day Name */}
                    <div className="w-1/3 md:w-1/4 text-white font-medium text-lg">
                      {getDayName(day.day)}
                    </div>
                    
                    {/* Icon & Target Prediction */}
                    <div className="w-1/3 flex items-center justify-center gap-3">
                      <Thermometer className="w-5 h-5 text-blue-400" />
                      <span className="text-white font-bold text-lg">{day.temp}°</span>
                    </div>
                    
                    {/* High/Low Range */}
                    <div className="w-1/3 flex justify-end items-center gap-4 text-slate-400 font-medium">
                      <div className="flex flex-col items-end md:flex-row md:items-center gap-1 md:gap-3">
                        <span>L: {day.low}°</span>
                        <div className="hidden md:block w-24 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                          {/* Fake range bar for aesthetics */}
                          <div className="h-full bg-linear-to-r from-blue-500 to-orange-400 w-full opacity-70"></div>
                        </div>
                        <span className="text-white">H: {day.high}°</span>
                      </div>
                    </div>
                  </div>

                  {/* Expanded Details Section */}
                  {expandedDay === day.day && (
                    <div className="p-4 bg-slate-800/80 border-t border-slate-700/50 grid grid-cols-3 gap-4 animate-in slide-in-from-top-2 duration-200">
                      <div className="flex flex-col items-center p-3 bg-slate-700/30 rounded-xl">
                        <Wind className="w-5 h-5 text-slate-400 mb-1" />
                        <span className="text-xs text-slate-500 uppercase font-semibold">Wind</span>
                        <span className="text-white font-medium">{day.wind !== undefined ? `${day.wind} km/h` : '--'}</span>
                      </div>
                      
                      <div className="flex flex-col items-center p-3 bg-slate-700/30 rounded-xl">
                        <Cloud className="w-5 h-5 text-slate-400 mb-1" />
                        <span className="text-xs text-slate-500 uppercase font-semibold">Clouds</span>
                        <span className="text-white font-medium">{day.clouds !== undefined ? `${day.clouds}%` : '--'}</span>
                      </div>
                      
                      <div className="flex flex-col items-center p-3 bg-slate-700/30 rounded-xl">
                        <CloudRain className="w-5 h-5 text-slate-400 mb-1" />
                        <span className="text-xs text-slate-500 uppercase font-semibold">Precip</span>
                        <span className="text-white font-medium">{day.precip !== undefined ? `${day.precip} mm` : '--'}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

// Reusable micro-component for the metrics
function MetricCard({ icon, title, value }) {
  return (
    <div className="bg-slate-800/50 rounded-2xl p-4 md:p-6 border border-slate-700/50 flex flex-col md:flex-row items-start md:items-center gap-3 md:gap-4">
      <div className="p-3 bg-slate-700/50 rounded-xl text-blue-400 shrink-0">
        {icon}
      </div>
      <div>
        <p className="text-xs md:text-sm font-medium text-slate-400">{title}</p>
        <p className="text-lg md:text-xl font-semibold text-white">{value}</p>
      </div>
    </div>
  );
}