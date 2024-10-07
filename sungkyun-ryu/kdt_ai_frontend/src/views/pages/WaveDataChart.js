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
import RealtimeStaticChart from "../../components/RealtimeStaticChart";

export default function MainPage() {


    const message = useRef('')
    const [staticMessage, setStaticMessage] = useState('');
    const [assetName, setAssetName] = useState('')
    const [assetId, setAssetId] = useState('');
    const [isChartsVisible, setIsChartsVisible] = useState(false)
    const [dataType, setDataType] = useState('')
    const [specData, setSpecData] = useState(null)
    const [waveData, setWaveData] = useState(null)
    const availableIds = ids_assets[assetName] || [];
    let chartName = useRef('')
    const sDate = useRef('')
    const sTime = useRef('')

    const sUnixTime = () => {
        if (sDate.current && sTime.current) {
            const sDateTimeString = `${sDate.current}T${sTime.current}`;
            const sDateTime = new Date(sDateTimeString);
            if (!isNaN(sDateTime.getTime())) {
                return Math.floor(sDateTime.getTime() / 1000);
            } else {
                console.error("Invalid date/time string:", sDateTimeString);
                return null
            }
        }
    };

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

        if (assetId && assetName && sTime.current && sDate.current) {
            console.log('핸들서브밋', assetId, assetName, sTime, sDate, sUnixTime())

            e.preventDefault();
            const param = {
                'asset_id': assetId,
                "created_at": sUnixTime()
            }
            PostDataFetch(param, 'http://192.168.0.126:8080/charts/detailAll')
                .then(result => {
                    console.log('result', result)
                    const newSpecData = {
                        created_at: new Date(result[0].created_at * 1000).toLocaleString('en-KR', {
                            timeZone: 'Asia/Seoul',
                        }),
                        x: result[0].spectrum_x,
                        y: result[0].spectrum_y,
                        z: result[0].spectrum_z,
                    }
                    const newWaveData = {
                        created_at: new Date(result[0].created_at * 1000).toLocaleString('en-KR', {
                            timeZone: 'Asia/Seoul',
                        }),
                        x: result[0].waveform_x,
                        y: result[0].waveform_y,
                        z: result[0].waveform_z,
                    }
                    setWaveData([newWaveData]);
                    setSpecData([newSpecData])
                })
            setIsChartsVisible(true)
        }
        else {
            alert('Please Choose All the Options')
        }
    }

    return (
        <>
            <CheckLogin>
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

                                <div className="flex items-center pr-24">
                                    <label htmlFor='date' className="text-xl block font-bold">Date & Time :</label> &nbsp;
                                    <input type='date' id='sDate' name='sDate' className=' text-lg'
                                        onChange={(e) => (sDate.current = e.target.value)}
                                    />
                                    <input type='time' id='sTime' name='sTime' className=' text-lg'
                                        onChange={(e) => (sTime.current = e.target.value)}
                                    />
                                </div>

                            </div>

                            <Button type='submit'
                                name='START'
                                text={'bg-blue-500 text-white px-4 py-1 rounded-full shadow hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 mr-4'}
                            />
                        </form>
                    </div>
                </div>

                <div className='flex w-full'>

                    <div className="p-10 w-1/2">

                        {waveData && (     
                            <>
                                <div className='flex'>
                                    <span className='font-bold text-lg'>{`Observed at: ${waveData[0].created_at}`}</span>
                                </div>
                                    <RealtimeStaticChart data={waveData[0].x} axis='x' colour='#6f63b9' datatype="Waveform" />
                                    <RealtimeStaticChart data={waveData[0].y} axis='y' colour='#ADD8E6' datatype="Waveform" />
                                    <RealtimeStaticChart data={waveData[0].z} axis='z' colour='#aadd80' datatype="Waveform" />
                            </>
                        ) }
                    </div>
                    <div className="p-10 w-1/2">

                        {specData && (
                                                        <>
                                <div className='flex'>
                                    <span className='font-bold text-lg'>{`Observed at: ${waveData[0].created_at}`}</span>
                                </div>
                                    <RealtimeStaticChart data={specData[0].x} axis='x' colour='#6f63b9' datatype="Spectrum" />
                                    <RealtimeStaticChart data={specData[0].y} axis='y' colour='#ADD8E6' datatype="Spectrum" />
                                    <RealtimeStaticChart data={specData[0].z} axis='z' colour='#aadd80' datatype="Spectrum" />
                            </>
                        )}
                    </div>
                </div>
            </CheckLogin>
        </>
    )
}

