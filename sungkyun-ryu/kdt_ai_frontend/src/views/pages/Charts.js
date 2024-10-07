import Chart from "../../components/Chart";
import Selection from "../../components/Selection";
import Nav from "../layouts/Nav";
import Button from "../../components/Button";
import Table from "../../components/Table";
import { PostDataFetch } from "../../functions/DataFetching";
import { generateUniqueId, addingUniqueId } from "../../functions/ProduceRows";
import { useState, useEffect, useRef } from "react";
import CheckBx from "../../components/CheckBx";
import { ProduceCols } from "../../functions/ProduceCols";
import { asset_names, col_names, ids_assets } from "../../components/Assets";
import Modal from "../../components/Modal";
import Toast from "../../components/Toast";
import { useLocation } from "react-router-dom";
import CheckLogin from "../../components/CheckLogin";


export default function Charts() {

  const [asset, setAsset] = useState('');
  const [assetId, setAssetId] = useState('');
  const [defaultDate, setDefaultDate] = useState('');
  const [defaultTime, setDefaultTime] = useState('');
  const sDate = useRef('')
  const sTime = useRef('')
  const eDate = useRef('')
  const eTime = useRef('')

  const [tableRows, setTableRows] = useState([]);
  const checkboxesRef = useRef([]);
  const [showCheckboxes, setShowCheckboxes] = useState(true);
  const [isTableVisible, setIsTableVisible] = useState(false);
  const [isChartsVisible, setIsChartsVisible] = useState(false);
  const [columns, setColumns] = useState([])

  const verticalLines = useRef([]);
  const commentRef = useRef('');
  const [modalOpen, setModalOpen] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const availableIds = ids_assets[asset] || [];
  const location = useLocation();
  const { bookmarkParams } = location.state || {};

  const eUnixTime = () => {
    if (eDate.current && eTime.current) {
      const eDateTimeString = `${eDate.current}T${eTime.current}`;
      const eDateTime = new Date(eDateTimeString);
      if (!isNaN(eDateTime.getTime())) {
        return Math.floor(eDateTime.getTime() / 1000);
      } else {
        console.error("Invalid date/time string:", eDateTimeString);
        return null;
      }
    }
  }

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
    else {
      return eUnixTime() - (7 * 24 * 60 * 60);
    }
  };

  const fetchChartData = async (params) => {
    try {
      const result = await PostDataFetch(params, 'http://192.168.0.126:8080/charts')

        .then(result => {
          const uniqueData = addingUniqueId(result);
          setTableRows(uniqueData);
          setColumns(ProduceCols(checkboxesRef.current));

          if (uniqueData.length > 0) {
            setIsTableVisible(true);
            setIsChartsVisible(true);
            setShowCheckboxes(false);
          } else {
            setIsTableVisible(false);
            setIsChartsVisible(false);
            setShowCheckboxes(true);
            alert('No data available');
          }

          return uniqueData;
        })
        .then(() => generateUniqueId.reset())
        .catch(error => {
          console.error('Error fetching data:', error);
        });
    } catch (error) {
      console.error('Error fetching chart data:', error);
      alert('Failed to fetch data.');
    }
  };


  const handleSubmit = async (e) => {

    e.preventDefault();

    setShowCheckboxes(false);

    if (asset === '') {
      alert('Please Select Asset Name')
    }

    if (asset !== '' && assetId === '') { alert('Please Select Asset Id') }

    if (checkboxesRef.current.length === 0) {
      alert('Please select at least one checkbox.');
      setShowCheckboxes(true);
      return;
    }
    
    const params = {
        "asset_id" : assetId,
        "start_at": sUnixTime(),
        "end_at": eUnixTime(),
        "cols": checkboxesRef.current,
      }

    fetchChartData(params)

    verticalLines.current = [];
  }

  const handleSelectionChange = (e) => {
    setAsset(e.target.value);
  };

  const handleIdChange = (e) => {
    setAssetId(e.target.value);
  };

  const handleChartClick = (params) => {
    if (params && params.componentType === 'series' && params.seriesType === 'line') {
      const xValue = params.data[0];

      if (verticalLines.current.length === 1) {
        verticalLines.current = [...verticalLines.current, xValue];
        setModalOpen(true);
      } else {
        verticalLines.current = [xValue]
        setToastMessage('Start Time saved successfully.');
        setShowToast(true);
      }
    }
  };
  
  const handleConfirm = () => {

    const params = {
      "asset_id": assetId,
      "start_at": verticalLines.current[0] / 1000,
      "end_at": verticalLines.current[1] / 1000,
      "cols": checkboxesRef.current,
      "bookmark_name": commentRef.current,
    }

    PostDataFetch(params, 'http://192.168.0.126:8080/charts/savebookmark')
    verticalLines.current = [];
    commentRef.current = '';
    setModalOpen(false);
  };

  const handleClose = () => {
    verticalLines.current = []
    setModalOpen(false);
  };

  const bookmarkAll = () => {
    verticalLines.current[0] = sUnixTime() * 1000
    verticalLines.current[1] = eUnixTime() * 1000
    setModalOpen(true);
  }

  useEffect(() => {
    const now = new Date();
    const nowDate = now.toISOString().split('T')[0];
    const nowTime = now.toTimeString().split(' ')[0].substring(0, 5);

    setDefaultDate(nowDate);
    setDefaultTime(nowTime);

    eDate.current = nowDate;
    eTime.current = nowTime;

  }, []);

  useEffect(() => {
  }, [verticalLines]);

  useEffect(() => {
    if (verticalLines.current.length > 1) { setModalOpen(true) }
  }, [verticalLines.current])

  useEffect(() => {
    if (verticalLines.current.length === 1) {
      setToastMessage('Start Time saved successfully.')
      setShowToast(true);
    }
  }, [verticalLines.current])

  useEffect(() => {
    if (bookmarkParams) {
      checkboxesRef.current = bookmarkParams.cols
      console.log('Received parameters:', bookmarkParams);
      fetchChartData(bookmarkParams);       
      console.log(bookmarkParams)
    }
  }, [bookmarkParams]);

  return (    
    <div>

      <header className='bg-black p-3'>
        <Nav />
      </header>
      <div className="flex justify-center items-center p-3">
        <div className=' w-5/6 justify-center items-center p-10 mb-10 border-b-4 border-gray-800 py-2'>
          <form className='flex justify-center items-center' onSubmit={handleSubmit}>
            <div className='w-auto'>
              <div className='flex justify-start items-center py-2 '>
                <label className="flex text-xl pr-24 font-bold items-center" > <span>Asset :</span> &nbsp;
                  <Selection choices={asset_names} text={'font-normal text-lg'}
                    func={handleSelectionChange} d_value={'Select Asset Name'}
                  />
                </label>

                <label className="flex text-xl font-bold items-center" > <span>Asset ID :</span> &nbsp;
                  <Selection choices={availableIds} text={'font-normal text-lg'}
                    func={handleIdChange} d_value={'Select Asset Id'}
                    selectedValue={assetId}
                  />
                </label>
              </div>

              <div className='flex justify-start items-center py-2'>
                <div className='flex justify-start '>
                  <div className="flex items-center pr-24">
                    <label htmlFor='date' className="text-xl block font-bold">Start :</label> &nbsp;
                    <input type='date' id='sDate' name='sDate' className=' text-lg'
                      onChange={(e) => (sDate.current = e.target.value)}
                    />
                    <input type='time' id='sTime' name='sTime' className=' text-lg'
                      onChange={(e) => (sTime.current = e.target.value)}
                    />
                  </div>

                  <div className="flex pr-20 items-center ">
                    <label htmlFor='time' className="text-xl block font-bold pl-1 ">End :</label>

                    <input type='date' id='eDate' name='eDate' className='text-lg' defaultValue={defaultDate}
                      onChange={(e) => (eDate.current = e.target.value)}
                    />
                    <input type='time' id='eTime' name='eTime' className=' text-lg' defaultValue={defaultTime}
                      onChange={(e) => (eTime.current = e.target.value)}
                    />
                  </div>

                </div>
                <div className='flex items-center'>
                  <Button type='submit' name='SUBMIT' text='bg-blue-500 text-white px-4 py-1 rounded-full shadow
                                           hover:bg-blue-600 focus:outline-none focus:ring-2
                                           focus:ring-blue-400 focus:ring-opacity-75'/>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>

      <div className='px-10 pb-24'>
        <div>
          <CheckBx text='font-bold text-xl' cols={col_names}
            isVisible={showCheckboxes} setIsVisible={setShowCheckboxes}
            ref={checkboxesRef}
          />
        </div>
        {isTableVisible && <Table rows={tableRows} columns={columns} text='mb-20' />}
        {isChartsVisible && <Button type='submit' name='Bookmark All' text='bg-blue-500 text-white px-4 py-1 rounded-full shadow
                                           hover:bg-blue-600 focus:outline-none focus:ring-2
                                           focus:ring-blue-400 focus:ring-opacity-75'
          func={bookmarkAll} />
        }
        {isChartsVisible && checkboxesRef.current.slice(0, 5).map((item) => (
          <div key={item} className='mb-5 border-b pb-10 border-gray-300'>
            <Chart key={item} cols={checkboxesRef.current} vis={item} sigData={tableRows}
              chartClickEvent={handleChartClick} />
            <Modal isOpen={modalOpen} onClose={handleClose} onConfirm={handleConfirm}
              text={<>
                Do you want to bookmark the data between <br />
                {new Date(verticalLines.current[0]).toLocaleString()} and {new Date(verticalLines.current[1]).toLocaleString()}?
              </>}
              ref={commentRef} />
            <Toast message={toastMessage} show={showToast} onClose={() => setShowToast(false)} />
          </div>
        ))}
      </div>
    </div>
  )
};