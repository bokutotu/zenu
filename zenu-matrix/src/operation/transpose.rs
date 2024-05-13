use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Repr},
};

impl<R: Repr, D: Device> Matrix<R, DimDyn, D> {
    pub fn transpose(&mut self) {
        let shape_stride = self.shape_stride();
        let transposed = shape_stride.transpose();
        self.update_shape_stride(transposed);
    }

    pub fn transpose_by_index(&mut self, index: &[usize]) {
        let shape_stride = self.shape_stride();
        let transposed = shape_stride.transpose_by_index(index);
        self.update_shape_stride(transposed);
    }

    pub fn transpose_by_index_new_matrix(
        &self,
        index: &[usize],
    ) -> Matrix<Owned<R::Item>, DimDyn, D> {
        let mut ref_mat = self.to_ref();
        ref_mat.transpose_by_index(index);
        ref_mat.to_default_stride()
    }

    pub fn transpose_swap_index(&mut self, a: usize, b: usize) {
        if a == b {
            panic!("Index must be different");
        }
        if a < b {
            return self.transpose_swap_index(b, a);
        }
        assert!(a < self.shape().len(), "Index out of range");
        assert!(b < self.shape().len(), "Index out of range");

        let shape_stride = self.shape_stride().swap_index(a, b);
        self.update_shape_stride(shape_stride);
    }

    pub fn transpose_swap_index_new_matrix(
        &self,
        a: usize,
        b: usize,
    ) -> Matrix<Owned<R::Item>, DimDyn, D> {
        if a == b {
            panic!("Index must be different");
        }
        if a < b {
            return self.transpose_swap_index_new_matrix(b, a);
        }
        let mut ref_mat = self.to_ref();
        ref_mat.transpose_swap_index(a, b);
        ref_mat.to_default_stride()
    }
}
#[cfg(test)]
mod transpose {
    use crate::{
        device::{Device, DeviceBase},
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    // #[test]
    fn transpose_2d<D: Device>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.transpose();
        assert_eq!(a.index_item([0, 0]), 1.);
        assert_eq!(a.index_item([0, 1]), 4.);
        assert_eq!(a.index_item([1, 0]), 2.);
        assert_eq!(a.index_item([1, 1]), 5.);
        assert_eq!(a.index_item([2, 0]), 3.);
        assert_eq!(a.index_item([2, 1]), 6.);
    }
    #[test]
    fn transpose_2d_cpu() {
        transpose_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn transpose_2d_cuda() {
        transpose_2d::<crate::device::nvidia::Nvidia>();
    }
}

#[cfg(test)]
mod transpose_inplace {
    // use crate::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};
    //
    // use super::TransposeInplace;

    use crate::{
        device::{Device, DeviceBase},
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    fn inplace_transpose_4d<D: Device>() {
        let mut input = vec![];
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    for l in 0..6 {
                        input.push((i * 1000 + j * 100 + k * 10 + l) as f32);
                    }
                }
            }
        }
        let input: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(input, [3, 4, 5, 6]);
        let output = input.transpose_by_index_new_matrix(&[1, 0, 2, 3]);
        let ans = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0,
            23.0, 24.0, 25.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1010.0, 1011.0, 1012.0, 1013.0,
            1014.0, 1015.0, 1020.0, 1021.0, 1022.0, 1023.0, 1024.0, 1025.0, 1030.0, 1031.0, 1032.0,
            1033.0, 1034.0, 1035.0, 1040.0, 1041.0, 1042.0, 1043.0, 1044.0, 1045.0, 2000.0, 2001.0,
            2002.0, 2003.0, 2004.0, 2005.0, 2010.0, 2011.0, 2012.0, 2013.0, 2014.0, 2015.0, 2020.0,
            2021.0, 2022.0, 2023.0, 2024.0, 2025.0, 2030.0, 2031.0, 2032.0, 2033.0, 2034.0, 2035.0,
            2040.0, 2041.0, 2042.0, 2043.0, 2044.0, 2045.0, 100.0, 101.0, 102.0, 103.0, 104.0,
            105.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 120.0, 121.0, 122.0, 123.0, 124.0,
            125.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 140.0, 141.0, 142.0, 143.0, 144.0,
            145.0, 1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1105.0, 1110.0, 1111.0, 1112.0, 1113.0,
            1114.0, 1115.0, 1120.0, 1121.0, 1122.0, 1123.0, 1124.0, 1125.0, 1130.0, 1131.0, 1132.0,
            1133.0, 1134.0, 1135.0, 1140.0, 1141.0, 1142.0, 1143.0, 1144.0, 1145.0, 2100.0, 2101.0,
            2102.0, 2103.0, 2104.0, 2105.0, 2110.0, 2111.0, 2112.0, 2113.0, 2114.0, 2115.0, 2120.0,
            2121.0, 2122.0, 2123.0, 2124.0, 2125.0, 2130.0, 2131.0, 2132.0, 2133.0, 2134.0, 2135.0,
            2140.0, 2141.0, 2142.0, 2143.0, 2144.0, 2145.0, 200.0, 201.0, 202.0, 203.0, 204.0,
            205.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 220.0, 221.0, 222.0, 223.0, 224.0,
            225.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 240.0, 241.0, 242.0, 243.0, 244.0,
            245.0, 1200.0, 1201.0, 1202.0, 1203.0, 1204.0, 1205.0, 1210.0, 1211.0, 1212.0, 1213.0,
            1214.0, 1215.0, 1220.0, 1221.0, 1222.0, 1223.0, 1224.0, 1225.0, 1230.0, 1231.0, 1232.0,
            1233.0, 1234.0, 1235.0, 1240.0, 1241.0, 1242.0, 1243.0, 1244.0, 1245.0, 2200.0, 2201.0,
            2202.0, 2203.0, 2204.0, 2205.0, 2210.0, 2211.0, 2212.0, 2213.0, 2214.0, 2215.0, 2220.0,
            2221.0, 2222.0, 2223.0, 2224.0, 2225.0, 2230.0, 2231.0, 2232.0, 2233.0, 2234.0, 2235.0,
            2240.0, 2241.0, 2242.0, 2243.0, 2244.0, 2245.0, 300.0, 301.0, 302.0, 303.0, 304.0,
            305.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 320.0, 321.0, 322.0, 323.0, 324.0,
            325.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 340.0, 341.0, 342.0, 343.0, 344.0,
            345.0, 1300.0, 1301.0, 1302.0, 1303.0, 1304.0, 1305.0, 1310.0, 1311.0, 1312.0, 1313.0,
            1314.0, 1315.0, 1320.0, 1321.0, 1322.0, 1323.0, 1324.0, 1325.0, 1330.0, 1331.0, 1332.0,
            1333.0, 1334.0, 1335.0, 1340.0, 1341.0, 1342.0, 1343.0, 1344.0, 1345.0, 2300.0, 2301.0,
            2302.0, 2303.0, 2304.0, 2305.0, 2310.0, 2311.0, 2312.0, 2313.0, 2314.0, 2315.0, 2320.0,
            2321.0, 2322.0, 2323.0, 2324.0, 2325.0, 2330.0, 2331.0, 2332.0, 2333.0, 2334.0, 2335.0,
            2340.0, 2341.0, 2342.0, 2343.0, 2344.0, 2345.0,
        ];
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(ans, [4, 3, 5, 6]);
        assert!((output - ans).asum() < 1e-6);
    }

    // #[test]
    fn swap_axis<D: Device>() {
        let input: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let output = input.transpose_swap_index_new_matrix(0, 1);
        let ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 4., 2., 5., 3., 6.], [3, 2]);
        assert!((output - ans).asum() < 1e-6);
    }
    #[test]
    fn inplace_transpose_4d_cpu() {
        inplace_transpose_4d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn inplace_transpose_4d_cuda() {
        inplace_transpose_4d::<crate::device::nvidia::Nvidia>();
    }

    // #[test]
    fn swap_axis_3d<D: Device>() {
        // 2, 3, 4
        let input: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            [2, 3, 4],
        );
        let output = input.transpose_swap_index_new_matrix(0, 1);
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 13., 14., 15., 16., 5., 6., 7., 8., 17., 18., 19., 20., 9., 10.,
                11., 12., 21., 22., 23., 24.,
            ],
            [3, 2, 4],
        );
        assert!((output - ans).asum() < 1e-6);
    }
    #[test]
    fn swap_axis_3d_cpu() {
        swap_axis_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn swap_axis_3d_cuda() {
        swap_axis_3d::<crate::device::nvidia::Nvidia>();
    }
}
