import ReactECharts from 'echarts-for-react';
import { useEffect, useRef, useState } from 'react';

export default function RealtimeChart({ data , axis, colour }) {

  const chartRef = useRef(null);
  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  const oneMinutesAgo = Date.now() - 1 * 60 * 1000;
  const threeSecondsAgo = Date.now() - 3 * 1000;
  const [zoomState, setZoomState] = useState({ start: 0, end: 100 });

  const transformData = (data) => {
    return data.map(item =>
      [item.time, item.value],
    )
  };
 
  const seriesData = [{
    name: axis,
    type: 'line',
    data: transformData(data.filter(item => 
      new Date(item.time).getTime() >= fiveMinutesAgo)),
      symbol: 'none',
      color: colour,
    }] 

  const options = {
    title: {
      text: `${axis} chart`,
      top: 'center',
      left: '0%'
    },
    tooltip: {
      trigger: 'axis',
    },
    grid: {
      left: '20%',
      right: '10%',
      bottom: '10%',
      top: '30%'
    },
    xAxis: {
      type: 'time',
      axisLabel: {
        show: false},
      axisTick: {
        show: false,
    },
    },
    yAxis: {
      type: 'value',
    },
    series: [...seriesData],
    dataZoom:
    {
      type: 'inside',
      start: zoomState.start,
      end: zoomState.end,
    }
  };

  useEffect(() => {
    const chartInstance = chartRef.current?.getEchartsInstance();

    if (chartInstance) {
      const updateZoomState = () => {
        const currentZoom = chartInstance.getOption().dataZoom[0];
        setZoomState({ start: currentZoom.start, end: currentZoom.end })
      }
      chartInstance.on('dataZoom', updateZoomState);

      return () => {
        chartInstance.off('dataZoom', updateZoomState);
      };
    }
  }, [zoomState]);

  return (
    <div>
      <ReactECharts option={options} ref={chartRef} />
    </div>
  )
}
