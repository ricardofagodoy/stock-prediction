google.charts.load('current', {'packages':['corechart']})

google.charts.setOnLoadCallback(async () => {

    // Fetch ticks from server
    const response = await fetch('tick')
    const ticks = await response.json()

    var data = google.visualization.arrayToDataTable(ticks, true);

    // Build candle chart
    var chart = new google.visualization.CandlestickChart(document.getElementById('chart_div'));

    chart.draw(data, {
        legend:'none',
        bar: { groupWidth: '100%' },
        candlestick: {
            fallingColor: { strokeWidth: 0, fill: '#a52714' }, // red
            risingColor: { strokeWidth: 0, fill: '#0f9d58' },   // green
        }
    })

    // Updates text with advice
    // TODO
    const action = document.getElementById('action')
    action.innerHTML = 'SELL'
    action.style.color = 'red'

    const percent = document.getElementById('percent')
    percent.innerHTML = '87%'
    percent.style.color = 'green'
})