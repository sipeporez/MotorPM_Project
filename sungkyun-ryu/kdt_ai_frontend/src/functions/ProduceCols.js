
export const ProduceCols = (cols) => {
    return [
        { field: 'local_time', headerName: 'time', width: 200 },
        { field: 'asset_id', headerName: 'asset_ID', width: 350 },
        ...cols.map((item) => ({
             field: item , headerName: item, width: 150
        }))
    ];
};

export const BookmarkCols = [
        { field: 'asset_id', headerName: 'asset_id', width: 400 },
        { field: 'cols', headerName: 'features', width: 400 },
        { field: 'startDate', headerName: 'start_date', width: 200 },
        { field: 'endDate', headerName: 'end_date', width: 200 },
        { field: 'bookmark_name', headerName: 'comment', width: 200 },
        { field: 'regidate', headerName: 'regidate', width: 200 },
    ]

export const AnomalyCols = [
    { field: 'asset_id', headerName: 'asset_id', width: 400 },
    { field: 'asset_name', headerName: 'asset_name', width: 400 },
    { field: 'created_at', headerName: 'time', width: 200 },
    { field: 'model_result', headerName: 'imbalance_status', width: 200 },
    { field: 'model_value', headerName: 'imbalnace_score', width: 200 },
]