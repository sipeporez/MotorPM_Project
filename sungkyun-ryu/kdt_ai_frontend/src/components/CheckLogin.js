import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Nav from '../views/layouts/Nav';

const CheckLogin = ({ children }) => {
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const token = sessionStorage.getItem('Token');
    const userid = sessionStorage.getItem('userid');
    
    if (!(token || userid)) {
      alert("Please log in first.");
      navigate('/login'); 
    } else {
      setLoading(false);  
    }
  }, [navigate]);

  if (loading) {

    return (
      <header className='bg-black p-3'>
        <Nav />
      </header>
    );
  }

  return <>{children}</>;
};

export default CheckLogin;
