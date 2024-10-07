import React from 'react'
import Nav from '../layouts/Nav'
import AnomalyTable from '../../components/BookmarkTable'
import axios from 'axios'
import { AnomalyCols } from '../../functions/ProduceCols'
import { useState, useEffect } from 'react'
import { addingUniqueId } from '../../functions/ProduceRows'
import CheckLogin from '../../components/CheckLogin'

export default function Anomaly() {

    const [anomalyColumns, setAnomalyColumns] = useState([]);
    const [anomalyRows, setAnomalyRows] = useState([]);
    const [isAnomalyTableVisible, setIsAnomalyTableVisible] = useState(false);
    const userid = sessionStorage.getItem('userid')

    const convertToKST = (timestamp) => {
        const date = new Date(timestamp * 1000); 
        const options = {
          timeZone: 'Asia/Seoul',
          year: 'numeric',
          month: 'numeric',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        };
        return date.toLocaleString('ko-KR', options);
      };

    const loadAnomaly = async () => {
        const token = sessionStorage.getItem('Token')
        try { 
            const response = await axios.get(
                'http://192.168.0.126:8080/anomaly', {
                    headers: {
                        'Authorization': `${token}`, 
                        'Content-Type': 'application/json', 
                    },
                })
                .then(result => {
                    const uniqueData = addingUniqueId(result.data);
                    setAnomalyRows(uniqueData);
                    setAnomalyColumns(AnomalyCols);

                    if (uniqueData.length > 0) {
                      setIsAnomalyTableVisible(true);            
                    } else {
                      setIsAnomalyTableVisible(false);                  
                      alert('No data available'); 
                    }        
                    return uniqueData; 
                  }                )          

        } catch (error) {
            console.log('loading anomaly error', error)
        }
    };   

    useEffect(() => {
        loadAnomaly(); 
    },[]);

    useEffect(() => {
        console.log('AnomalyRows',anomalyRows)
    }, [anomalyRows])

    return (
        <div>
        <CheckLogin>
            <header className='bg-black p-3'>
                <Nav />
            </header>
            <div className='px-10 pb-24'>
                
                <div className='flex justify-center mb-10 py-10 '>
                <span className='font bold text-5xl p-3 border-b-4 w-auto border-b-slate-600'>Anomaly List</span>
                </div>
                {isAnomalyTableVisible && <AnomalyTable columns={anomalyColumns} 
                                                            rows={anomalyRows}
                                                            text='mb-20'/>}
            </div>
            </CheckLogin>
        </div>
    )
}
