use crate::tlas::blas::blas_bvh::boundingbox::{surrounding_box, AABB};
use crate::tlas::blas::blas_bvh::HasBox;
use crate::INFINITY;

pub(super) struct BLASInfo
{
    bounding_box: AABB,
    blas_index: u8,
    matrices: Vec<glam::Affine3A>,
}

impl BLASInfo
{
    pub fn new(bounding_box: AABB, blas_index: u8, matrices: Vec<glam::Affine3A>) -> Self
    {
        Self {
            bounding_box,
            blas_index,
            matrices,
        }
    }
}

pub(super) enum TLASNodeType
{
    Leaf
    {
        blas_index: u8,
        matrix: glam::Affine3A,
        inv_matrix: glam::Affine3A,
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

impl HasBox for TLASNode
{
    fn get_box(&self) -> &AABB { &self.bounding_box }
}

impl TLASNode
{
    fn find_best_match(nodes: &[Box<TLASNode>], a_index: usize) -> usize
    {
        let a: &AABB = &nodes[a_index].bounding_box;

        let mut best_sa: f32 = INFINITY;
        let mut best_index: usize = usize::MAX;

        for i in 0..nodes.len()
        {
            if i == a_index
            {
                continue;
            }

            let b: &AABB = &nodes[i].bounding_box;

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

    pub fn generate_tlas(blas_info: &[BLASInfo]) -> Self
    {
        let mut nodes = blas_info
            .iter()
            .flat_map(|info| {
                info.matrices.iter().map(|&matrix| {
                    let bb = info.bounding_box.transform(&matrix);
                    let node = Self {
                        bounding_box: bb,
                        node_type: TLASNodeType::Leaf {
                            blas_index: info.blas_index,
                            matrix,
                            inv_matrix: matrix.inverse(),
                        },
                    };

                    Box::new(node)
                })
            })
            .collect::<Vec<_>>();

        let mut a: usize = 0;
        let mut b: usize = Self::find_best_match(&nodes, a);

        while nodes.len() > 1
        {
            let c: usize = Self::find_best_match(&nodes, b);
            if a == c
            {
                let node_a = nodes.swap_remove(a);
                let node_b = nodes.swap_remove(b);

                let new_node = Box::new(Self {
                    bounding_box: surrounding_box(&node_a.bounding_box, &node_b.bounding_box),
                    node_type: TLASNodeType::Branch { left: node_a, right: node_b },
                });

                a = nodes.len();
                nodes.push(new_node);
                b = Self::find_best_match(&nodes, a);
            }
            else
            {
                a = b;
                b = c;
            }
        }

        Box::into_inner(nodes.remove(0))
    }
}
