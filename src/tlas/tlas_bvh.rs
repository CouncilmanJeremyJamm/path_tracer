use crate::tlas::blas::blas_bvh::boundingbox::{surrounding_box, AABB};
use std::cmp::Ordering;

pub(super) struct BLASInfo
{
    bounding_box: AABB,
    blas_index: u8,
}

impl BLASInfo
{
    pub fn new(bounding_box: AABB, blas_index: u8) -> Self { Self { bounding_box, blas_index } }
    pub fn create_bounding_box(object_info: &[Self]) -> AABB
    {
        object_info
            .iter()
            .fold(AABB::identity(), |a: AABB, b: &BLASInfo| surrounding_box(&a, &b.bounding_box))
    }
}

pub(super) enum TLASNodeType
{
    Leaf
    {
        blas_index: u8
    },
    Branch
    {
        left: Box<TLASNode>, right: Box<TLASNode>
    },
}

pub(super) struct TLASNode
{
    pub bounding_box: AABB,
    pub node_type: TLASNodeType,
}

impl TLASNode
{
    pub fn generate_tlas(blas_info: &mut [BLASInfo]) -> Self
    {
        let blas_span: usize = blas_info.len();
        debug_assert!(blas_span > 0);

        if blas_span == 1
        {
            Self {
                bounding_box: blas_info[0].bounding_box,
                node_type: TLASNodeType::Leaf {
                    blas_index: blas_info[0].blas_index,
                },
            }
        }
        else
        {
            let bounding_box: AABB = BLASInfo::create_bounding_box(blas_info);

            //Find longest axis
            let box_length: glam::Vec3A = bounding_box.length();
            let max_length: f32 = box_length.max_element();

            let split_axis: u8 = if box_length.x == max_length
            {
                0u8
            }
            else if box_length.y == max_length
            {
                1u8
            }
            else
            {
                2u8
            };

            let comparator = |a: &BLASInfo, b: &BLASInfo| -> Ordering { a.bounding_box.compare(&b.bounding_box, split_axis) };
            blas_info.sort_unstable_by(comparator);

            let (left_info, right_info) = blas_info.split_at_mut(blas_span / 2);
            let (left, right) = rayon::join(|| Box::new(Self::generate_tlas(left_info)), || Box::new(Self::generate_tlas(right_info)));

            Self {
                bounding_box,
                node_type: TLASNodeType::Branch { left, right },
            }
        }
    }
}