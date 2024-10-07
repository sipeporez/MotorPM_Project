import axios from 'axios';

const PostDataFetch = async (params, url) => {
  try {
    console.log('datafetchparams', params)
    const token = sessionStorage.getItem('Token');   
    const response = await axios.post(url, params
      , { headers: { Authorization: token } }
    ); 
    return response.data

  } catch (error) {
    console.error("===> Error occured while fetching data", error)
  }
};

export  { PostDataFetch };
