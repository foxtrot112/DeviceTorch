#include "Numora.hpp"

inline void Numora::Matrix::store(u32 _row, u32 _col, f32 value)
{
    this->data[_row * cols + _col] = value;
}

f32 Numora::Matrix::fetch(u32 _row, u32 _col)
{
    return this->data[_row * cols + _col];
}

void Numora::Matrix::fill_random(f32 _range)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; ++j)
        {
            f32 min_value = -_range;
            f32 max_value = _range;
            f32 random_value = min_value + static_cast<f32>(rand()) / (static_cast<f32>(RAND_MAX / (max_value - min_value)));

            store(i, j, random_value);
        }
    }
}

void Numora::Matrix::debug_print()
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << fetch(i, j) << " ";
        }
        std::cout << "\n";
    }
}

inline Numora::Matrix Numora::Matrix::transpose()
{
    Matrix result(this->cols, this->rows);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.store(j, i, fetch(i, j));
        }
    }

    return result;
}

void Numora::Matrix::flush()
{
    delete[] data;
}

void Numora::Matrix::save_cache(std::__1::string path)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f)
        return;

    fwrite(&this->rows, sizeof(int), 1, f);
    fwrite(&this->cols, sizeof(int), 1, f);
    fwrite(this->data, sizeof(float), this->rows * this->cols, f);

    fclose(f);
}

void Numora::Matrix::load_cache(std::__1::string path)
{
    int _rows, _cols;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
    {
        std::cerr << "Failed to open file: " << path << "\n";
        return;
    }

    // Read shape
    fread(&_rows, sizeof(int), 1, f);
    fread(&_cols, sizeof(int), 1, f);

    // Allocate memory for matrix
    this->rows = rows;
    this->cols = cols;
    this->data = (float *)malloc(sizeof(float) * rows * cols);

    if (!this->data)
    {
        std::cerr << "Memory allocation failed\n";
        fclose(f);
        return;
    }

    // Read data
    size_t read_count = fread(this->data, sizeof(float), rows * cols, f);

    if (read_count != (size_t)(rows * cols))
    {
        std::cerr << "File read error or corrupted file\n";
    }

    fclose(f);
}
float Numora::Matrix::operator()(int row, int col)
{
    return this->fetch(row, col);
}

Numora::Matrix Numora::Matrix::operator*(const Numora::Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Number of columns of the first matrix must equal the number of rows of the second matrix for multiplication.");
    }

    Matrix result(rows, other.cols);

    for (u32 i = 0; i < rows; ++i)
    {
        for (u32 j = 0; j < other.cols; ++j)
        {
            f32 sum = 0.0f;

            for (u32 k = 0; k < cols; ++k)
            {
                sum += this->data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.store(i, j, sum);
        }
    }

    return result;
}
Numora::Matrix Numora::Matrix::operator*(const f32 &other) const
{
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.store(i, j, this->data[i * cols + j] * other);
        }
    }

    return result;
}

Numora::Matrix Numora::Matrix::operator+(const Numora::Matrix &other) const
{
    Matrix result(rows, cols);

    if (cols != other.cols || rows != other.rows)
    {
        throw std::runtime_error("Unmatched matrix sizes");
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.store(i, j, this->data[i * cols + j] + other.data[i * cols + j]);
        }
    }

    return result;
}

Numora::Matrix Numora::Matrix::operator-(const Numora::Matrix &other) const
{
    Matrix result(rows, cols);

    if (cols != other.cols || rows != other.rows)
    {
        throw std::runtime_error("Unmatched Matrix sizes");
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.store(i, j, this->data[i * cols + j] - other.data[i * cols + j]);
        }
    }

    return result;
}

Numora::Matrix Numora::Matrix::outer_product(Numora::Matrix _other) const
{
    if (this->cols != 1 || _other.cols != 1)
    {
        throw std::runtime_error("Invalid Coordinate Vectors");
    }

    return *this * _other.transpose();
}

Numora::Matrix Numora::Matrix::element_wise(Numora::Matrix _other) const
{
    if (this->cols != _other.cols)
    {
        throw std::runtime_error("Unmatch vector size");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.store(i, j, this->data[i * cols + j] * _other.data[i * cols + j]);
        }
    }

    return result;
}