import numpy as np

# 1. HÀM LOAD DỮ LIỆU
def load_data(file_path):
    """
    Đọc dữ liệu từ CSV sử dụng np.genfromtxt.
    - dtype=None: Để Numpy tự động đoán kiểu dữ liệu (int, float, string).
    - names=True: Để lấy dòng đầu tiên làm tên cột (header).
    - encoding='utf-8': Xử lý ký tự đặc biệt.
    """
    try:
        data = np.genfromtxt(file_path, delimiter=',', dtype=None, names=True, encoding='utf-8')
        return data
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

# 2. HÀM LẤY THÔNG TIN CƠ BẢN (Thay thế df.info() và df.describe())
def get_column_stats(data, column_name):
    """
    Trả về thống kê mô tả của một cột.
    """
    col_data = data[column_name]
    
    # Kiểm tra kiểu dữ liệu của cột
    if np.issubdtype(col_data.dtype, np.number):
        # Nếu là số: Tính mean, median, std, min, max
        # Cần loại bỏ nan trước khi tính toán
        clean_data = col_data[~np.isnan(col_data)]
        stats = {
            "type": "numeric",
            "mean": np.mean(clean_data),
            "median": np.median(clean_data),
            "std": np.std(clean_data),
            "min": np.min(clean_data),
            "max": np.max(clean_data),
            "missing_count": np.isnan(col_data).sum()
        }
    else:
        # Nếu là chữ (categorical): Đếm số lượng giá trị unique
        unique, counts = np.unique(col_data, return_counts=True)
        # Sắp xếp theo số lượng giảm dần
        sorted_indices = np.argsort(-counts)
        top_values = dict(zip(unique[sorted_indices][:5], counts[sorted_indices][:5])) # Top 5 giá trị
        
        # Missing value trong Numpy string thường là chuỗi rỗng '' hoặc b'' hoặc 'nan'
        missing_mask = (col_data == '') | (col_data == 'nan') | (col_data == 'NaN')
        stats = {
            "type": "categorical",
            "unique_count": len(unique),
            "top_values": top_values,
            "missing_count": np.sum(missing_mask)
        }
    
    return stats

# 3. HÀM XỬ LÝ MISSING VALUES (Imputation)
def impute_missing(data, column_name, strategy='mean', fill_value=None):
    """
    Điền giá trị thiếu.
    - Numeric: strategy = 'mean' hoặc 'median'
    - Categorical: strategy = 'mode' (giá trị xuất hiện nhiều nhất) hoặc 'constant'
    """
    col_data = data[column_name].copy()
    
    if np.issubdtype(col_data.dtype, np.number):
        mask = np.isnan(col_data)
        if strategy == 'mean':
            fill_val = np.nanmean(col_data)
        elif strategy == 'median':
            fill_val = np.nanmedian(col_data)
        else:
            fill_val = fill_value
        
        col_data[mask] = fill_val
        
    else:
        # Với string, missing thường là chuỗi rỗng hoặc 'nan'
        mask = (col_data == '') | (col_data == 'nan')
        if strategy == 'mode':
            unique, counts = np.unique(col_data[~mask], return_counts=True)
            fill_val = unique[np.argmax(counts)]
        else:
            fill_val = fill_value if fill_value else 'Unknown'
            
        col_data[mask] = fill_val
        
    return col_data

# 4. HÀM ENCODING (Biến đổi Categorical -> Số)
def encode_label(col_data):
    """
    Label Encoding: Chuyển đổi chuỗi sang số nguyên (0, 1, 2...).
    Trả về: (mảng đã mã hóa, từ điển mapping)
    """
    unique_vals, indices = np.unique(col_data, return_inverse=True)
    mapping = {val: i for i, val in enumerate(unique_vals)}
    return indices, mapping

# 5. HÀM CHUẨN HÓA (Min-Max Scaling)
def min_max_scale(col_data):
    """Đưa dữ liệu về đoạn [0, 1]"""
    min_val = np.min(col_data)
    max_val = np.max(col_data)
    if max_val - min_val == 0:
        return np.zeros_like(col_data)
    return (col_data - min_val) / (max_val - min_val)

# 6. HÀM XỬ LÝ CỘT KINH NGHIỆM (Experience)
def clean_experience(col_data):
    """
    Chuyển đổi cột experience từ string ('>20', '<1', '15') sang số (float).
    Missing values sẽ được để là NaN.
    """
    # Tạo mảng kết quả, mặc định là NaN
    result = np.full(col_data.shape, np.nan, dtype=float)
    
    # Xử lý các trường hợp đặc biệt
    # numpy.char là module xử lý chuỗi mạnh mẽ của Numpy
    col_str = col_data.astype(str)
    
    # Case '>20' -> 21
    mask_gt20 = (col_str == '>20')
    result[mask_gt20] = 21.0
    
    # Case '<1' -> 0
    mask_lt1 = (col_str == '<1')
    result[mask_lt1] = 0.0
    
    # Các trường hợp số thông thường ('1', '5', '10'...)
    # Lọc những giá trị không phải >20, <1 và không phải nan/rỗng
    mask_normal = ~mask_gt20 & ~mask_lt1 & (col_str != 'nan') & (col_str != '')
    
    # Dùng try-catch trong vectorization là không thể, nên ta dùng np.where
    # Tuy nhiên để an toàn, ta loop nhẹ hoặc dùng kỹ thuật ép kiểu
    # Ở đây ta dùng np.char.replace để xóa ký tự thừa nếu có và ép kiểu
    temp_vals = col_str[mask_normal]
    try:
        result[mask_normal] = temp_vals.astype(float)
    except ValueError:
        pass # Nếu lỗi thì vẫn để NaN
        
    return result

# 7. HÀM XỬ LÝ QUY MÔ CÔNG TY (Company Size)
def clean_company_size(col_data):
    """
    Chuyển đổi khoảng nhân viên thành số thứ tự (Ordinal Encoding).
    <10      -> 0
    10/49    -> 1
    50-99    -> 2
    100-500  -> 3
    500-999  -> 4
    1000-4999-> 5
    5000-9999-> 6
    10000+   -> 7
    """
    mapping = {
        '<10': 0,
        '10/49': 1,
        '50-99': 2,
        '100-500': 3,
        '500-999': 4,
        '1000-4999': 5,
        '5000-9999': 6,
        '10000+': 7
    }
    
    # Tạo mảng kết quả mặc định -1 (tượng trưng cho Missing/Unknown)
    result = np.full(col_data.shape, -1, dtype=int)
    
    col_str = col_data.astype(str)
    
    for key, val in mapping.items():
        result[col_str == key] = val
        
    return result

# 8. HÀM XỬ LÝ HỌC VẤN (Ordinal Encoding)
def clean_education_level(col_data):
    """
    Primary School -> 0
    High School    -> 1
    Graduate       -> 2
    Masters        -> 3
    Phd            -> 4
    """
    mapping = {
        'Primary School': 0,
        'High School': 1,
        'Graduate': 2,
        'Masters': 3,
        'Phd': 4
    }
    
    result = np.full(col_data.shape, -1, dtype=int) # -1 là missing
    col_str = col_data.astype(str)
    
    for key, val in mapping.items():
        result[col_str == key] = val
        
    return result

# 9. CHUẨN HÓA Z-SCORE (Standardization)
def standard_scale(col_data):
    """(x - mean) / std"""
    mean = np.nanmean(col_data)
    std = np.nanstd(col_data)
    if std == 0:
        return np.zeros_like(col_data)
    return (col_data - mean) / std

# 10. KIỂM ĐỊNH CHI-SQUARE (Thống kê)
def chi_square_test_numpy(feature_col, target_col):
    """
    Tính giá trị Chi-square để kiểm tra mối quan hệ giữa 2 biến categorical.
    """
    # 1. Tạo bảng chéo (Contingency Table)
    cats_feat = np.unique(feature_col)
    cats_target = np.unique(target_col)
    
    observed = np.zeros((len(cats_feat), len(cats_target)))
    
    for i, f_val in enumerate(cats_feat):
        for j, t_val in enumerate(cats_target):
            observed[i, j] = np.sum((feature_col == f_val) & (target_col == t_val))
            
    # 2. Tính giá trị kỳ vọng (Expected)
    row_sums = np.sum(observed, axis=1)
    col_sums = np.sum(observed, axis=0)
    total = np.sum(observed)
    
    expected = np.outer(row_sums, col_sums) / total
    
    # 3. Tính Chi-square statistic
    # Thêm epsilon nhỏ để tránh chia cho 0
    chi2 = np.sum((observed - expected)**2 / (expected + 1e-9))
    
    return chi2, len(cats_feat)-1  # Trả về chi2 và bậc tự do (với target binary df = (r-1)(2-1))

# 11. XÁC ĐỊNH VÀ LOẠI BỎ NGOẠI LAI (Outliers)
def remove_outliers_iqr(X, y, col_idx):
    """Loại bỏ outliers sử dụng IQR"""
    data = X[:, col_idx]
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (data >= lower) & (data <= upper)
    return X[mask], y[mask]