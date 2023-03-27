use std::cmp::Ordering;

use ambassador::Delegate;
use id_arena::{Arena, Id};
use rayon::prelude::ParallelSliceMut;

use crate::tlas::tlas_bvh::blas::blas_bvh::boundingbox::{surrounding_box, HasBox, AABB};
use crate::tlas::tlas_bvh::blas::primitive::Triangle;

#[macro_use]
pub mod boundingbox;

const DESIRED_BINS: usize = 64;

const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 2.0;

pub(crate) struct PrimitiveInfo
{
    bounding_box: AABB,
    primitive_id: Id<Triangle>,
}

impl PrimitiveInfo
{
    pub fn new(bounding_box: AABB, primitive_id: Id<Triangle>) -> Self { Self { bounding_box, primitive_id } }

    pub fn create_bounding_box(object_info: &[Self]) -> AABB
    {
        object_info
            .iter()
            .fold(AABB::identity(), |a: AABB, b: &PrimitiveInfo| surrounding_box(&a, &b.bounding_box))
    }
}

pub(crate) enum BLASNodeType
{
    LeafSingle
    {
        primitive_id: Id<Triangle>
    },
    Leaf
    {
        primitive_ids: Box<[Id<Triangle>]>
    },
    Branch
    {
        left: Id<BLASNode>, right: Id<BLASNode>
    },
}

#[derive(Delegate)]
#[delegate(HasBox, target = "bounding_box")]
pub(crate) struct BLASNode
{
    pub(crate) bounding_box: AABB,
    pub(crate) node_type: BLASNodeType,
}

impl BLASNode
{
    pub(in crate::tlas) fn generate_blas(blas_arena: &mut Arena<BLASNode>, object_info: &mut [PrimitiveInfo], last_split_axis: u8) -> Id<Self>
    {
        let object_span: usize = object_info.len();
        debug_assert!(object_span > 0);

        if object_span == 1
        {
            blas_arena.alloc(Self {
                bounding_box: object_info[0].bounding_box,
                node_type: BLASNodeType::LeafSingle {
                    primitive_id: object_info[0].primitive_id,
                },
            })
        }
        else
        {
            let bounding_box: AABB = PrimitiveInfo::create_bounding_box(object_info);
            let bb_sa: f32 = bounding_box.surface_area();

            //Find longest axis
            let split_axis: u8 = bounding_box.longest_axis();

            //Sort, for splitting along longest axis
            //Indices are still correctly sorted if the parent node sorted along the same axis
            if split_axis != last_split_axis
            {
                let comparator = |a: &PrimitiveInfo, b: &PrimitiveInfo| -> Ordering { a.bounding_box.compare(&b.bounding_box, split_axis) };
                glidesort::sort_by(object_info, comparator);
                // object_info.par_sort_unstable_by(comparator);
            }

            let bin_size: usize = (object_span / DESIRED_BINS).max(1);
            let num_bins: usize = (object_span / bin_size) - 1;

            let (best_split, best_split_sah): (usize, f32) = (0..num_bins)
                .into_iter()
                .map(|i: usize| {
                    let j: usize = (i + 1) * bin_size;

                    let l_box: AABB = PrimitiveInfo::create_bounding_box(&object_info[..j]);
                    let r_box: AABB = PrimitiveInfo::create_bounding_box(&object_info[j..]);

                    let sah: f32 = TRAVERSAL_COST
                        + ((j as f32) * l_box.surface_area() + ((object_span - j) as f32) * r_box.surface_area()) * INTERSECTION_COST / bb_sa;

                    (j, sah)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            let no_split_sah = INTERSECTION_COST * (object_span as f32);

            if no_split_sah < best_split_sah
            {
                blas_arena.alloc(Self {
                    bounding_box,
                    node_type: BLASNodeType::Leaf {
                        primitive_ids: object_info.iter().map(|a| a.primitive_id).collect::<Vec<_>>().into_boxed_slice(),
                    },
                })
            }
            else
            {
                let (left_info, right_info): (&mut [PrimitiveInfo], &mut [PrimitiveInfo]) = object_info.split_at_mut(best_split);

                let left: Id<BLASNode> = Self::generate_blas(blas_arena, left_info, split_axis);
                let right: Id<BLASNode> = Self::generate_blas(blas_arena, right_info, split_axis);

                blas_arena.alloc(Self {
                    bounding_box,
                    node_type: BLASNodeType::Branch { left, right },
                })
            }
        }
    }
}
