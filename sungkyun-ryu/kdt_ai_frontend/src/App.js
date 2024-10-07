import { Routes, Route, BrowserRouter, Navigate } from 'react-router-dom';

import MainPage from './views/pages/MainPage';
import LoginPage from './views/pages/LoginPage';
import Charts from './views/pages/Charts';
import Bookmark from './views/pages/Bookmark';
import WaveDataChart from './views/pages/WaveDataChart';
import Anomaly from './views/pages/Anomaly';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path='/login' element={<LoginPage />} />
        <Route path='/home' element={<MainPage />} />
        <Route path='/histwave' element={<WaveDataChart/>} />
        <Route path='/charts' element={<Charts />} />
        <Route path='/bookmark' element={<Bookmark />} />
        <Route path='/anomaly' element={<Anomaly />}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
