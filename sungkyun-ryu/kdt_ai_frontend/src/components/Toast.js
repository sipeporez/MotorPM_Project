import React, { useEffect } from 'react';

const Toast = ({ message, show, onClose }) => {
  useEffect(() => {
    if (show) {
      const timer = setTimeout(() => {
        onClose();
      }, 900); 

      return () => clearTimeout(timer);
    }
  }, [show, onClose]);

  return (
    show && (
      <div className="fixed bottom-5 right-5 bg-green-500 text-white p-4 rounded shadow-lg transition-opacity duration-300">
        {message}
      </div>
    )
  );
};

export default Toast;
