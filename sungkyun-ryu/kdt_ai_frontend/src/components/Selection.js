import { useState, useEffect } from "react"

export default function Selection({choices, func, text, d_value, selectedValue}) {

  const [options, setOptions] = useState([]);

  useEffect(() => {
    setOptions(choices)
  }, [choices])

  return (
    <div>
      <select value={selectedValue} onChange= {func} className={text}>
      <option value= '' selected>{d_value}</option>
        {options.map((option, index) => (
          <option key={index} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  )
}
