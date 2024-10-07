import ReactECharts from 'echarts-for-react';
import React from 'react'

export default function Gauge({opt, val}) {

  const volt_options = {
    tooltip: {
      formatter: '{a} <br/>{b} : {c}%'
    },
    series: [
      {
        name: 'Voltage',
        type: 'gauge',
        min: 1.5, 
        max: 3.5, 
        detail: {
          formatter: function(value) {
                    return value.toFixed(2); 
                } ,
          textStyle: {
            fontSize: 15,  
          }
        },
        data: [
          {
            value: val,
            name: 'Voltage'
          }
        ]
      }
    ]
  };

  const temp_options = {
    series: [
      {
        type: 'gauge',
        center: ['50%', '60%'],
        startAngle: 200,
        endAngle: -20,
        radius: '60%',
        min: 0,
        max: 60,
        splitNumber: 12,
        itemStyle: {
          color: '#FFAB91'
        },
        progress: {
          show: true,
          width: 30
        },
        pointer: {
          show: false
        },
        axisLine: {
          lineStyle: {
            width: 30
          }
        },
        axisTick: {
          distance: -45,
          splitNumber: 5,
          lineStyle: {
            width: 2,
            color: '#999'
          }
        },
        splitLine: {
          distance: -52,
          length: 14,
          lineStyle: {
            width: 3,
            color: '#999'
          }
        },
        axisLabel: {
          distance: -20,
          color: '#999',
          fontSize: 20
        },
        anchor: {
          show: false
        },
        title: {
          show: false
        },
        detail: {
          valueAnimation: true,
          width: '60%',
          lineHeight: 40,
          borderRadius: 8,
          offsetCenter: [0, '-15%'],
          fontSize: 30,
          fontWeight: 'bolder',
          formatter: '{value} °C',
          color: 'inherit'
        },
        data: [
          {
            value: val 
          }
        ]
      }
    ]
  };




  return (
    <div>
      <ReactECharts option={(opt === 'temp') ? temp_options : volt_options} />
    </div>
  )
}
