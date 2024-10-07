import pandas as pd
# pip install SQLAlchemy
from sqlalchemy import create_engine 

# 데이터베이스 연결 설정
db_config = {
    'user': 'root',
    'password': 'tiger',
    'host': 'localhost',
    'database': 'pm_data',
}

# SQLAlchemy 엔진 생성
connection_string = (
    f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@"
    f"{db_config['host']}/{db_config['database']}"
)
engine = create_engine(connection_string)

# SQL 쿼리
query = "SELECT DISTINCT asset_id FROM ics_asset_mst"

# 데이터프레임으로 변환
asset_data = pd.read_sql(query, engine)

for _, row in asset_data.iterrows():
    asset_id = row['asset_id']

    # 쿼리 작성
    sql_query = f"""
    SELECT 
        s.asset_id, s.created_at, s.created_at_datetime, s.temperature, s.rms_x, s.rms_y, s.rms_z, s.rms_xyz, s.vel_rms_x, s.vel_rms_y, s.vel_rms_z, s.vel_rms_xyz, s.skewness_x, s.skewness_y, s.skewness_z, s.vel_skewness_x, s.vel_skewness_y, s.vel_skewness_z, s.kurtosis_x, s.kurtosis_y, s.kurtosis_z, s.vel_kurtosis_x, s.vel_kurtosis_y, s.vel_kurtosis_z, s.crest_factor_x, s.crest_factor_y, s.crest_factor_z, s.vel_crest_factor_x, s.vel_crest_factor_y, s.vel_crest_factor_z, s.peak_x, s.peak_y, s.peak_z, s.vel_peak_x, s.vel_peak_y, s.vel_peak_z, s.peak2peak_x, s.peak2peak_y, s.peak2peak_z, s.vel_peak2peak_x, s.vel_peak2peak_y, s.vel_peak2peak_z,  -- sig 테이블의 모든 열
        h.time, h.imbalance_health, h.misalignment_health, h.looseness_health, h.bearing_health, h.asset_health   -- hist 테이블의 모든 열
    FROM 
        ics_asset_sigdata s
    JOIN 
       ics_asset_status_hist h 
    ON 
        s.asset_id = h.asset_id  -- asset_id 기준으로 조인
    AND 
        DATE(FROM_UNIXTIME(s.created_at)) + INTERVAL 1 DAY = DATE(h.time)  -- sig의 날짜가 hist의 날짜의 하루 전이 맞도록 조인
    WHERE 
        FROM_UNIXTIME(s.created_at) >= h.time - INTERVAL 1 DAY
    AND 
        FROM_UNIXTIME(s.created_at) < h.time
    AND 
        s.asset_id = '{asset_id}'
    AND 
        h.asset_health IS NOT NULL
    ORDER BY 
        s.created_at;
    """
    
    # 쿼리 실행 및 데이터프레임으로 변환
    df = pd.read_sql(sql_query, engine)
    
    # 
    # CSV 파일 이름을 asset_id로 인덱스 없이 저장
    df.to_csv(f'{asset_id}_data.csv', index=False)
