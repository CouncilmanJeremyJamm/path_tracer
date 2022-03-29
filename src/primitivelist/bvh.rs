use std::cmp::Ordering;

use rayon::prelude::ParallelSliceMut;

use crate::primitivelist::bvh::boundingbox::{surrounding_box, AABB};

pub mod boundingbox;

const DESIRED_BINS: usize = 64;

const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 10.0;

pub(super) struct PrimitiveInfo
{
    bounding_box: AABB,
    primitive_index: u32,
}

impl PrimitiveInfo
{
    pub fn new(bounding_box: AABB, primitive_index: u32) -> Self
    {
        Self {
            bounding_box,
            primitive_index,
        }
    }
}

fn create_bounding_box(object_info: &[PrimitiveInfo]) -> AABB
{
    object_info
        .iter()
        .fold(AABB::identity(), |a: AABB, b: &PrimitiveInfo| surrounding_box(&a, &b.bounding_box))
}

#[derive(Clone)]
pub(super) enum NodeType
{
    Leaf
    {
        primitive_indices: Vec<u32>, num_objects: u32
    },
    Branch
    {
        left: Box<BVHNode>, right: Box<BVHNode>
    },
}

#[derive(Clone)]
pub(super) struct BVHNode
{
    pub(crate) bounding_box: AABB,
    pub(crate) node_type: NodeType,
}

impl BVHNode
{
    #[must_use]
    pub(super) fn generate_bvh(object_info: &mut [PrimitiveInfo], last_split_axis: u8) -> Self
    {
        let object_span: usize = object_info.len();
        debug_assert!(object_span > 0);

        if object_span == 1
        {
            Self {
                bounding_box: object_info[0].bounding_box,
                node_type: NodeType::Leaf {
                    primitive_indices: vec![object_info[0].primitive_index],
                    num_objects: 1,
                },
            }
        }
        else
        {
            let bounding_box: AABB = create_bounding_box(object_info);
            let bb_sa: f32 = bounding_box.surface_area();

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

            //Sort, for splitting along longest axis
            //Indices are still correctly sorted if the parent node sorted along the same axis
            //Don't sort twice unnecessarily
            if split_axis != last_split_axis
            {
                let comparator = |a: &PrimitiveInfo, b: &PrimitiveInfo| -> Ordering { a.bounding_box.compare(&b.bounding_box, split_axis) };
                object_info.par_sort_unstable_by(comparator);
            }

            let bin_size: usize = (object_span / DESIRED_BINS).max(1);
            let num_bins: usize = (object_span / bin_size) - 1;

            let (best_split, best_split_sah): (usize, f32) = (0..num_bins)
                .into_iter()
                .map(|i: usize| {
                    let j: usize = (i + 1) * bin_size;

                    let l_box: AABB = create_bounding_box(&object_info[..j]);
                    let r_box: AABB = create_bounding_box(&object_info[j..]);

                    let sah: f32 = TRAVERSAL_COST
                        + ((j as f32) * l_box.surface_area() + ((object_span - j) as f32) * r_box.surface_area()) * INTERSECTION_COST / bb_sa;

                    (j, sah)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let no_split_sah = INTERSECTION_COST * (object_span as f32);

            if no_split_sah < best_split_sah
            {
                Self {
                    bounding_box,
                    node_type: NodeType::Leaf {
                        primitive_indices: object_info.iter().map(|a| a.primitive_index).collect(),
                        num_objects: object_span as u32,
                    },
                }
            }
            else
            {
                let (left_indices, right_indices): (&mut [PrimitiveInfo], &mut [PrimitiveInfo]) = object_info.split_at_mut(best_split);

                let (left, right): (Box<BVHNode>, Box<BVHNode>) = rayon::join(
                    || Box::new(Self::generate_bvh(left_indices, split_axis)),
                    || Box::new(Self::generate_bvh(right_indices, split_axis)),
                );

                Self {
                    bounding_box,
                    node_type: NodeType::Branch { left, right },
                }
            }
        }
    }
}
