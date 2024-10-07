
import { useState, useEffect, forwardRef } from "react"
import Button from "./Button";

const CheckBx = forwardRef(function CheckBx({cols, isVisible, setIsVisible}, ref) {

  const [localSelectedCheckboxes, setLocalSelectedCheckboxes] = useState([]);

  const handleToggleVisibility = () => {
    setIsVisible(prev => !prev);
  };

  const handleCheckboxChange = (label) => {

    setLocalSelectedCheckboxes(prev => {
      if (prev.includes(label)) {
        return prev.filter(item => item !== label);
      } else {
        return [...prev, label];
      }
    });
  };

  const handleSelectAll = () => {
    if (localSelectedCheckboxes.length === cols.length) {
      setLocalSelectedCheckboxes([]);
    } else {
      setLocalSelectedCheckboxes(cols);    }
  };

  useEffect(() => {
    if(ref) {
    ref.current = localSelectedCheckboxes;
  } else {
    console.log('ref===> ', ref)
  }
  },[localSelectedCheckboxes])

  return (
    <div>

      <Button text={'bg-blue-500 text-white px-4 py-1 my-3 rounded-full shadow hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 '}
        func={handleToggleVisibility}
        name={isVisible ? 'Hide Checkbox' : 'Show Checkbox'} />

      {isVisible && (
        <div>
          <div className='mt-10'>
            <Button text={'px-4 py-1 border border-gray-300 rounded '}
              func={handleSelectAll}
              name={'Select All'} />
          </div>

          <div className="mt-4 mb-4 grid gap-4 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
            {cols.map((label, index) => (
              <div key={index} className="flex items-center p-2 border border-gray-300 rounded cursor-pointer"
                onClick={() => handleCheckboxChange(label)}>
                <input type="checkbox" id={`checkbox-${index}`} className="mr-2"
                  checked={localSelectedCheckboxes.includes(label)}
                  onChange={() => handleCheckboxChange(label)} />
                <label htmlFor={`checkbox-${index}`}
                  className="flex cursor-pointer">{label}</label>
              </div>
            ))
            }
          </div>
        </div>
      )}
    </div>
  )
}
);

export default CheckBx;
