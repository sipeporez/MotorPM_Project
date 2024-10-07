import ReactECharts from 'echarts-for-react';
import { useEffect, useRef, useState } from 'react';

export default function Chart({ cols, vis, sigData, chartClickEvent }) {

  const chartRef = useRef(null);

  const [selectedFeatures, setSelectedFeatures] = useState(cols.reduce((acc, series) => {
      acc[series] = series === vis;
      return acc;
    }, {}));

  const [zoomState, setZoomState] = useState({ start: 0, end: 100 });

  const transformData = (sigData, feature) => {
    return sigData.map(item =>
      [item.created_at * 1000, item[feature]],
    )
  };

  const handleLegendClick = (params) => {
    setSelectedFeatures(prev => ({
      ...prev,
      [params.name]: !prev[params.name],
    }));
  };

  const seriesData = cols.map(feature => ({
    name: feature,
    type: 'line',
    data: transformData(sigData, feature),
    symbol: 'circle',
    symbolSize: 3,
    emphasis: {
      itemStyle: {
        color: '#FF0000',
        symbolSize: 50,
      },
    },
  })) 

  const options = {
    title: {
      text: `${vis} chart`,
      top: 'center',
      left: '0%'
    },
    tooltip: {
      trigger: 'axis',
    },
    legend: sigData ? {
      data: cols,
      selected: selectedFeatures,
      top: '5%',
      left: 'center',
      bottom: '0%',
    } : undefined,
    grid: {
      left: '15%',
      right: '10%',
      bottom: '10%',
      top: '30%'
    },
    xAxis: {
      type: 'time',
      axisLabel: {
        formatter: (value) => {
          const date = new Date(value);
          return `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:00`;
        },
      },
    },
    yAxis: {
      type: 'value',
    },
    series: [
      ...seriesData,
    ],
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
  }, [chartClickEvent]);

  useEffect(() => {
    const chartInstance = chartRef.current?.getEchartsInstance();
    setZoomState({ start: 0, end: 100 })
    setSelectedFeatures(cols.reduce((acc, series) => {
      acc[series] = series === vis;
      return acc;
    }, {}))
    if (chartInstance) {
      chartInstance.on('legendselectchanged', handleLegendClick);
      chartInstance.on('click', chartClickEvent);
      return () => {
        chartInstance.off('legendselectchanged', handleLegendClick);
        chartInstance.off('click', chartClickEvent);
      };
    }
  }, [sigData]);

  return (
    <div>
      <ReactECharts option={options} ref={chartRef} />
    </div>
  )
}
