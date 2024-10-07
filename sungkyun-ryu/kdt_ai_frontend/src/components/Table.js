import { DataGrid } from '@mui/x-data-grid';
import Paper from '@mui/material/Paper';
import { useState, useEffect } from 'react';


export default function Table({ rows, columns, text}) {

    return (
        <div>
            <Paper sx={{
                height: 400, width: '100%',
                border: '4px solid black',
            }} className={text}>
                <DataGrid
                    rows={rows}
                    columns={columns}
                    pageSizeOptions={[5, 10]}
                    sx={{
                        border: 0,
                        '& .MuiDataGrid-columnHeaders': {
                            fontWeight: 'bold',
                            fontSize: '1.2rem',
                            color: 'black',
                        }
                    }}
                />
            </Paper>
        </div>
    )
}

