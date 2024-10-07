import ReactECharts from 'echarts-for-react';
import { useEffect, useRef, useState } from 'react';
import { spec_freq } from './Assets';

export default function RealtimeStaticChart({ data, axis, colour, datatype }) {

    console.log('realtimedatatype', datatype)
    const chartRef = useRef(null);
    const [zoomState, setZoomState] = useState({ start: 0, end: 100 });

    const transformData = (data) => {
        let sdata;

        if (datatype === "Spectrum") {
            sdata = data.map((value, index) =>
                [0.78125 * index, value]
            );
        } else {
            sdata = data.map((value, index) =>
                [index, value]
            );
        }
        return sdata;
    }

    const seriesData = [{
        name: axis,
        type: 'line',
        data: transformData(data),
        symbol: 'none',
        color: colour,
    }]

    const options = {
        title: {
            text: datatype === 'Spectrum' ? `spectrum_${axis}` : `waveform__${axis}`,
            top: 'center',
            left: '0%'
        },
        tooltip: datatype === 'Spectrum' ? {
            trigger: 'axis',
        } : {
            trigger: 'item',
            formatter: function (params) {
                if (params.length > 0 && params[0].data) {
                return `${params[0].data[1]}`;
            }
            return 'No Data'; },
        },

        grid: {
            left: '20%',
            right: '10%',
            bottom: '10%',
            top: '30%'
        },
        xAxis:
            (datatype === "Spectrum") ? {
                type: 'value',
                name: 'Frequency (Hz)',
                min: 0,
                max: 1600,
            } : {
                type: 'time',
                axisLabel: {
                    show: false
                },
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

    console.log('transformdata', transformData(data))

    return (
        <div>
            <ReactECharts option={options} ref={chartRef} />
        </div>
    )
}
