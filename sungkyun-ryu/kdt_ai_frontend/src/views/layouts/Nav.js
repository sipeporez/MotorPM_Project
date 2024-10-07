import Button from "../../components/Button";
import { useNavigate } from "react-router-dom";


export default function Nav() {

    const navigate = useNavigate();

    const logout = () => {
        sessionStorage.removeItem('Token');
        sessionStorage.removeItem('userid');
        navigate('/login');
    }

    return (
        <>
            <div className="p-5">
                <h1 className='flex justify-between items-center'>
                    <div className='flex'>
                    <a href='/home' className= 'mr-10 text-5xl text-white'>
                        PREDICTIVE MAINTENANCE 
                    </a>
                    <a href='/histwave' className= 'mr-10 text-2xl text-white'>
                        WaveData
                    </a>
                    <a href='/charts' className= 'mr-10 text-2xl text-white'>
                        SigData
                    </a>
                    <a href='/bookmark' className= 'mr-10 text-2xl text-white'>
                        Bookmark 
                    </a>               
                    <a href='/anomaly' className= 'mr-10 text-2xl text-white'>
                        Anomaly 
                    </a>             
                    </div>         
                    <div className="items-center justify-center">
                        <Button name={sessionStorage.getItem("Token") ? "LOGOUT" : "LOGIN" }
                                func = {logout}
                                text={"bg-white text-black text-xl font-bold px-3 py-1 rounded-full shadow hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75"}/>
                    </div>
                </h1>
            </div>
        </>
    )
}
