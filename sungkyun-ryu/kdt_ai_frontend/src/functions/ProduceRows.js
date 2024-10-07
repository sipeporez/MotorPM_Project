const generateUniqueId = (() => {
    let id = 1;
    return {
        next: () => id++,
        reset: () => { id = 1; }
    };
})();

const formatUnixToLocalTime = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
};

const addingUniqueId = (rows) => {
    if (Array.isArray(rows)) {
        return rows.map((row) => ({
            ...row,
            local_time: formatUnixToLocalTime(row.created_at),
            id: generateUniqueId.next(),
        }));
    } else return [];
};

export { addingUniqueId, generateUniqueId }; 