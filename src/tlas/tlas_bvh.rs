use ambassador::Delegate;
use id_arena::{Arena, Id};

use blas::blas_bvh::boundingbox::HasBox;
use blas::blas_bvh::boundingbox::{surrounding_box, AABB};
use blas::BLAS;

use crate::INFINITY;

#[macro_use]
pub mod blas;

pub(super) struct BLASInfo
{
    bounding_box: AABB,
    blas_id: Id<BLAS>,
    matrices: Vec<glam::Affine3A>,
}

impl BLASInfo
{
    pub fn new(bounding_box: AABB, blas_id: Id<BLAS>, matrices: Vec<glam::Affine3A>) -> Self
    {
        Self {
            bounding_box,
            blas_id,
            matrices,
        }
    }
}

pub(super) enum TLASNodeType
{
    Leaf
    {
        blas_id: Id<BLAS>,
        matrix: glam::Affine3A,
        inv_matrix: glam::Affine3A,
    },
    Branch
    {
        left: Id<TLASNode>, right: Id<TLASNode>
    },
}

#[derive(Delegate)]
#[delegate(HasBox, target = "bounding_box")]
pub(super) struct TLASNode
{
    pub bounding_box: AABB,
    pub node_type: TLASNodeType,
}

impl TLASNode
{
    fn find_best_match(tlas_arena: &Arena<Self>, nodes: &[Id<Self>], a_index: usize) -> usize
    {
        let a: &AABB = &tlas_arena[nodes[a_index]].bounding_box;

        let mut best_sa: f32 = INFINITY;
        let mut best_index: usize = usize::MAX;

        for i in 0..nodes.len()
        {
            if i == a_index
            {
                continue;
            }

            let b: &AABB = &tlas_arena[nodes[i]].bounding_box;

            let bounding_box: AABB = surrounding_box(a, b);
            let sa: f32 = bounding_box.surface_area();

            if sa < best_sa
            {
                best_index = i;
                best_sa = sa;
            }
        }

        best_index
    }

    pub fn generate_tlas(tlas_arena: &mut Arena<Self>, blas_info: &[BLASInfo]) -> Id<Self>
    {
        // Create vec of node ids with a minimum capacity (assuming 1 instance of every BLAS)
        let mut nodes: Vec<Id<Self>> = Vec::with_capacity(blas_info.len());

        for info in blas_info
        {
            for matrix in &info.matrices
            {
                let bb: AABB = info.bounding_box.transform(matrix);
                nodes.push(tlas_arena.alloc(Self {
                    bounding_box: bb,
                    node_type: TLASNodeType::Leaf {
                        blas_id: info.blas_id,
                        matrix: *matrix,
                        inv_matrix: matrix.inverse(),
                    },
                }));
            }
        }

        let mut a: usize = 0;
        let mut b: usize = Self::find_best_match(tlas_arena, &nodes, a);

        while nodes.len() > 1
        {
            let c: usize = Self::find_best_match(tlas_arena, &nodes, b);
            if a == c
            {
                let (node_1, node_2): (Id<Self>, Id<Self>) = if a > b
                {
                    (nodes.swap_remove(a), nodes.swap_remove(b))
                }
                else
                {
                    (nodes.swap_remove(b), nodes.swap_remove(a))
                };

                a = nodes.len();
                nodes.push(tlas_arena.alloc(Self {
                    bounding_box: surrounding_box(&tlas_arena[node_1].bounding_box, &tlas_arena[node_2].bounding_box),
                    node_type: TLASNodeType::Branch { left: node_1, right: node_2 },
                }));
                b = Self::find_best_match(tlas_arena, &nodes, a);
            }
            else
            {
                a = b;
                b = c;
            }
        }

        nodes.remove(0)
    }
}
