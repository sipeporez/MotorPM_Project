import Nav from "../layouts/Nav";
import Button from "../../components/Button";
import Selection from "../../components/Selection";
import Gauge from "../../components/Gauge";
import { useState, useRef, useEffect } from "react";
import WebSocketProvider from "../../components/WebSocketProvider";
import { asset_names, ids_assets } from "../../components/Assets";
import { PostDataFetch } from "../../functions/DataFetching";
import WebSocketStatic from "../../components/WebSocketStatic";
import CheckLogin from "../../components/CheckLogin";
import HealthGauge from "../../components/HealthGauge";

export default function MainPage() {


  const message = useRef('')
  const [staticMessage, setStaticMessage] = useState('');
  const [assetName, setAssetName] = useState('')

  const [assetId, setAssetId] = useState('');
  const [isChartsVisible, setIsChartsVisible] = useState(false)

  const [dataType, setDataType] = useState('')
  const availableIds = ids_assets[assetName] || [];
  const [temp, setTemp] = useState('')
  const [volt, setVolt] = useState('')

  let chartName = useRef('')
  const [health, setHealth] = useState(null)

  const handleNameChange = (e) => {
    setAssetName(e.target.value)
    setAssetId('')
  }

  const handleIdChange = (e) => {
    setAssetId(e.target.value);
    
  };

  const handleTypeChange = (e) => {
    setDataType(e.target.value)
  }

  const handleSubmit = (e) => {
    
    if (assetId && assetName && dataType) {

    e.preventDefault();

    const param = { 'asset_id': assetId }
    PostDataFetch(param, 'http://192.168.0.126:8080/tempvolt')
    .then(result => {    
      setVolt(result.voltage);
      setTemp(result.temperature)
    })

    if(dataType === "SPECTRUM") {chartName.current ='Spectrum'}

    else {chartName.current ='Waveform'}

    setStaticMessage(assetId + ', STATIC, ' + dataType);
    setIsChartsVisible(true)
    }
  
  else {
    alert('Please Choose All the Options')
  }
  }

  console.log('chartvis', isChartsVisible)
  console.log('staticmessage', staticMessage)
  useEffect(() => {
    console.log('HEALTH', health)
  }, [health] )

  return (
    <>
      <header className='bg-black p-3'>
        <Nav />
      </header>
      <div className='flex justify-center p-3'>
        <div className='border-4 rounded-full border-black w-auto flex justify-between'>

          <form className='flex items-center justify-between py-5 px-8 w-full' onSubmit={handleSubmit}>
            <div className='mr-10 flex'>
              <div>
              <label className="flex text-xl font-bold py-3 pr-8 items-center" > <span>Asset :</span> &nbsp;
                <Selection choices={asset_names} text={'font-normal text-lg'}
                  func={handleNameChange} d_value={'Please Select Asset Name'}
                />
              </label>
              </div>

              <div > 
              <label className="flex text-xl font-bold py-3 pr-8 items-center" > <span>Asset ID :</span> &nbsp;
                <Selection choices={availableIds} text={'font-normal text-lg'}
                  func={handleIdChange} d_value={'Please Select Asset Id '}
                  selectedValue={assetId} />
              </label>
              
              </div>
              <div>
              <label className="flex text-xl font-bold py-3 pr-8 items-center"> <span>Data Type :</span> &nbsp;
                <Selection choices={['SPECTRUM', 'WAVEFORM']} text={'font-normal text-lg'}
                  func={handleTypeChange} d_value={'Chart Type '} />
              </label>
              </div>
            </div>

            <Button type='submit' 
                    name='START'
                    text={'bg-blue-500 text-white px-4 py-1 rounded-full shadow hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 mr-4'} 
                      />
          </form>
        </div>
      </div>

      <div className="flex w-full p-10 ">

        <div className="w-2/3 flex flex-col justify-center items-center">
          
          {isChartsVisible &&  
          <span className='font-bold text-3xl border-b-4 border-black'>
           { `RealTime ${chartName.current} Charts`}
          </span>}         

          <WebSocketStatic message={staticMessage} datatype={chartName.current} sethealth={setHealth}/> 
       
        </div>

<div className="flex flex-col w-1/3">
<div className='h-1/2 w-full flex flex-col justify-center items-center'>
<div className='h-1/2 w-1/2 flex justify-center ' >
          
            {isChartsVisible && health && (
            <div className="flex flex-col w-full">
            <div className='pb-16 flex justify-center'>
            <span className='font-bold text-3xl border-b-4 border-black'>
              Imbalance Status
            </span>
            </div>
            <div className={`font-bold text-4xl border-4 border-white flex justify-center items-center py-5 px-10 rounded-full m-10
                        ${(health > 0.5) ?  'bg-blue-600' :
                        (health > .25)? 'bg-yellow-400':
                         'bg-red-600' } text-white`}>
              {(health > 0.5)  ?  'HEALTHY' :
              (health > .25)? 'INSPECT': 
              'CRITICAL' }
            </div>
            </div>)}

          </div>
          <div className="h-1/2 w-full pb-32 ">
          {isChartsVisible && health &&
           <HealthGauge val={health}/>}
          </div>
          </div>

        <div className='flex full justify-start'>          
          <div className='h-1/2 w-1/2' >
            {isChartsVisible && (
            <>
            <div className='pb-16 flex justify-center'>
            <span className='font-bold text-3xl border-b-4 border-black'>
              Temperature
            </span>
            </div>
            <Gauge opt={'temp'} val={temp}/>
            </>)}
          </div>
          <div className='h-1/2 w-1/2'>
            {isChartsVisible && (
              <>
              <div className='pb-16 flex justify-center'>
                <span className='font-bold text-3xl border-b-4 border-black '>
              Sensor Voltage
            </span>
            </div>
            <Gauge opt={'volt'} val={volt} /></>)}
          </div>
          </div>
        </div>



        </div>
    </>
  )
}













// //back up 
// import Nav from "../layouts/Nav";
// import Button from "../../components/Button";
// import Selection from "../../components/Selection";
// import Gauge from "../../components/Gauge";
// import { useState, useRef, useEffect } from "react";
// import WebSocketProvider from "../../components/WebSocketProvider";
// import { asset_names, ids_assets } from "../../components/Assets";
// import { PostDataFetch } from "../../functions/DataFetching";
// import RealtimeChart from "../../components/RealtimeChart";
// import WebSocketStatic from "../../components/WebSocketStatic";
// import Charts from "./Charts";

// export default function MainPage() {

//   const [message, setMessage] = useState('');
//   const [staticMessage, setStaticMessage] = useState('');
//   const [assetName, setAssetName] = useState('')
//   const [assetId, setAssetId] = useState('');
//   const [isChartsVisible, setIsChartsVisible] = useState(false)
//   const [chartStyle, setChartStyle] = useState('')
//   const availableIds = ids_assets[assetName] || [];
//   const [temp, setTemp] = useState('')
//   const [volt, setVolt] = useState('')
//   const [staticData, setStaticData] = useState({})
//   const [dataOn, setDataOn] = useState(false)
//   const handleNameChange = (e) => {
//     setAssetName(e.target.value)
//     setAssetId('')
//   }

//   const handleIdChange = (e) => {
//     setAssetId(e.target.value);
    
//   };

//   const handleStyleChange = (e) => {
//     setChartStyle(e.target.value)
//     console.log(chartStyle)
//   }

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     setDataOn(prev => !prev)
//     const param = { 'asset_id': assetId }
//     PostDataFetch(param, 'http://192.168.0.126:8080/tempvolt')
//     .then(result => {    
//       setVolt(result.voltage);
//       setTemp(result.temperature)
//     })
//     if (chartStyle === 'DYNAMIC') { setMessage(assetId + ', ' + chartStyle + ', spec'); }
//     else { setStaticMessage(assetId + ', ' + chartStyle + ', spec'); }
//     setIsChartsVisible(true)
//   }

//   console.log('chartvis', isChartsVisible)

//   return (
//     <>
//       <header className='bg-black p-3'>
//         <Nav />
//       </header>
//       <div className='flex justify-center p-3'>
//         <div className='border-4 rounded-full border-black w-auto flex justify-between'>

//           <form className='flex items-center justify-between py-5 px-8 w-full' onSubmit={handleSubmit}>
//             <div className='flex mr-10'>
//               <label className="flex text-xl font-bold py-3 pr-8 items-center" > <span>Asset :</span> &nbsp;
//                 <Selection choices={asset_names} text={'font-normal text-lg'}
//                   func={handleNameChange} d_value={'Please Select Asset Name'}
//                 />
//               </label>

//               <label className="flex text-xl font-bold py-3 pr-8 items-center" > <span>Asset ID :</span> &nbsp;
//                 <Selection choices={availableIds} text={'font-normal text-lg'}
//                   func={handleIdChange} d_value={'Please Select Asset Id '}
//                   selectedValue={assetId} />
//               </label>

//               <label className="flex text-xl font-bold items-center" > <span>Chart Style :</span> &nbsp;
//                 <Selection choices={['DYNAMIC', 'STATIC']} text={'font-normal text-lg'}
//                   func={handleStyleChange} d_value={'Chart Style '} />
//               </label>
//             </div>

//             <Button type='submit' 
//                     name={(dataOn === false) ? 'START' : 'STOP'}
//                     text={(dataOn === false) ? 'bg-blue-500 text-white px-4 py-1 rounded-full shadow hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 mr-4' 
//                                             : 'bg-red-500 text-white px-4 py-1 rounded-full shadow hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-opacity-75 mr-4'}
//                       />
//           </form>
//         </div>
//       </div>

//       <div className="flex w-full p-10 ">

//         <div className="w-2/3 flex flex-col justify-center items-center">
          
//           {isChartsVisible && 
//           <span className='font-bold text-3xl border-b-4 border-black'>
//            { chartStyle === 'DYNAMIC' ? 'RealTime Dynamic Waveform Charts': 'RealTime Static Waveform Charts'}
//           </span>}         
//           {/* { chartStyle === "DYNAMIC" ? <WebSocketProvider message={message} />    
//           : <WebSocketStatic message={staticMessage} dataon={dataOn} style={chartStyle} />       }       */}
//           {/* <WebSocketStatic message={staticMessage} dataon={dataOn} style={chartStyle} />             */}
//           <WebSocketProvider message={message} />  
//         </div>
//         <div className='w-1/3 justify-start'>
//           <div className='h-1/2' >
//             {isChartsVisible && (
//             <>
//             <div className='pb-16'><span className='font-bold text-3xl border-b-4 border-black'>
//               Temperature
//             </span></div>
//             <Gauge opt={'temp'} val={temp}/>
//             </>)}
//           </div>
//           <div className='h-1/2 mt-20'>
//             {isChartsVisible && (
//               <>
//               <div className='pb-16'><span className='font-bold text-3xl border-b-4 border-black '>
//               Voltage
//             </span></div>
//             <Gauge opt={'volt'} val={volt} /></>)}
//           </div>
//         </div>
//       </div>

//     </>
//   )
// }