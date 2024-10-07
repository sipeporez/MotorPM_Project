import ReactECharts from 'echarts-for-react';
import React from 'react'

export default function HealthGauge({val}) {

    const option = {
        series: [
          {
            type: 'gauge',
            startAngle: 180,
            endAngle: 0,
            center: ['50%', '75%'],
            radius: '80%',
            min: 0,
            max: 1,
            splitNumber: 2,
            axisLine: {
              lineStyle: {
                width: 6,
                color: [
                  [0.25, '#FF6E76'],
                  [0.5, '#FDDD60'],
                  [0.75, '#58D9F9'],
                  [1, '#58D9F9']
                ]
              }
            },
            pointer: {
              icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
              length: '12%',
              width: 20,
              offsetCenter: [0, '-60%'],
              itemStyle: {
                color: 'auto'
              }
            },
            axisTick: {
              length: 12,
              lineStyle: {
                color: 'auto',
                width: 2
              }
            },
            splitLine: {
              length: 20,
              lineStyle: {
                color: 'auto',
                width: 5
              }
            },
            axisLabel: {
              color: '#464646',
              fontSize: 20,
              distance: -60,
              rotate: 'tangential',
              formatter: function (value) {
                if (value > .5) {
                  return 'Healthy';
                } else if (value <= .5 && value >= .25) {
                  return 'Inspect';
                } else 
                  return 'Critical';
              }
            },
            title: {
              offsetCenter: [0, '-10%'],
              fontSize: 20
            },
            detail: {
              fontSize: 30,
              offsetCenter: [0, '-35%'],
              valueAnimation: true,
              formatter: function (value) {
                return Math.round(value * 100) + '';
              },
              color: 'inherit'
            },
            data: [
              {
                value: val,
                name: 'Score'
              }
            ]
          }
        ]
      };  

  return (
    <div>
      <ReactECharts option={option} />
    </div>
  )
}
