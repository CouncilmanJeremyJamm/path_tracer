use std::ops::Deref;

use wgpu::include_wgsl;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::{ASPECT_RATIO, IMAGE_HEIGHT, IMAGE_WIDTH};

struct Texture
{
    texture: wgpu::Texture,
}

impl Texture
{
    pub fn new(device: &wgpu::Device, size: wgpu::Extent3d, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self
    {
        let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
        });

        Self { texture }
    }

    pub fn view(&self) -> wgpu::TextureView { self.texture.create_view(&wgpu::TextureViewDescriptor::default()) }
}

impl Deref for Texture
{
    type Target = wgpu::Texture;

    fn deref(&self) -> &Self::Target { &self.texture }
}

pub struct State
{
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub(crate) size: PhysicalSize<u32>,

    accumulate_pipeline: wgpu::ComputePipeline,
    velocity_pipeline: wgpu::ComputePipeline,
    compute_pipeline: wgpu::ComputePipeline,

    accumulate_bind_group: wgpu::BindGroup,
    velocity_bind_group: wgpu::BindGroup,
    compute_bind_group: wgpu::BindGroup,

    input_texture: Texture,
    accumulation_texture: Texture,
    output_texture: Texture,
    position_texture: Texture,
    id_texture: Texture,

    render_bundle: wgpu::RenderBundle,

    viewport_origin: glam::Vec2,
    viewport_size: glam::Vec2,
}

// impl<'a> State<'a>
impl State
{
    // Creating some of the wgpu types requires async code
    pub(crate) async fn new(window: &Window) -> State
    {
        let size: PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface: wgpu::Surface = unsafe { instance.create_surface(window) };
        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits {
                        max_push_constant_size: std::mem::size_of::<glam::Mat4>() as u32,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let config: wgpu::SurfaceConfiguration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: dbg!(surface.get_supported_formats(&adapter)[0]),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        };

        surface.configure(&device, &config);

        let texture_size: wgpu::Extent3d = wgpu::Extent3d {
            width: IMAGE_WIDTH as u32,
            height: IMAGE_HEIGHT as u32,
            depth_or_array_layers: 1,
        };

        let position_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let id_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::R32Uint,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let accumulation_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let input_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let output_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );

        let velocity_texture: Texture = Texture::new(
            &device,
            texture_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING,
        );

        // RENDER PIPELINE
        let sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });

        let bind_group_layout: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accumulation_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader: wgpu::ShaderModule = device.create_shader_module(include_wgsl!("shaders/shader.wgsl"));

        let render_pipeline_layout: wgpu::PipelineLayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline: wgpu::RenderPipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            multiview: None,
        });

        let mut encoder: wgpu::RenderBundleEncoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: &[Some(config.format)],
            depth_stencil: None,
            sample_count: 1,
            multiview: None,
        });

        encoder.set_pipeline(&render_pipeline);
        encoder.set_bind_group(0, &bind_group, &[]);
        encoder.draw(0..3, 0..1);

        let render_bundle: wgpu::RenderBundle = encoder.finish(&wgpu::RenderBundleDescriptor::default());

        // Velocity
        let velocity_shader: wgpu::ShaderModule = device.create_shader_module(include_wgsl!("shaders/velocity.wgsl"));

        let velocity_bgl: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Velocity Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let velocity_bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity Bind Group"),
            layout: &velocity_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&position_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_texture.view()),
                },
            ],
        });

        let velocity_pl: wgpu::PipelineLayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Velocity Pipeline Layout"),
            bind_group_layouts: &[&velocity_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..(std::mem::size_of::<glam::Mat4>() as u32),
            }],
        });

        let velocity_pipeline: wgpu::ComputePipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Velocity Pipeline"),
            layout: Some(&velocity_pl),
            module: &velocity_shader,
            entry_point: "main",
        });

        // Accumulate
        let accumulate_shader: wgpu::ShaderModule = device.create_shader_module(include_wgsl!("shaders/accumulate.wgsl"));

        let accumulate_bgl: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Accumulate Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let accumulate_bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Accumulate Bind Group"),
            layout: &accumulate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accumulation_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view()),
                },
            ],
        });

        let accumulate_pl: wgpu::PipelineLayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Accumulate Pipeline Layout"),
            bind_group_layouts: &[&accumulate_bgl],
            push_constant_ranges: &[],
        });

        let accumulate_pipeline: wgpu::ComputePipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Accumulate Pipeline"),
            layout: Some(&accumulate_pl),
            module: &accumulate_shader,
            entry_point: "main",
        });

        // COMPUTE PIPELINE
        let compute_shader: wgpu::ShaderModule = device.create_shader_module(include_wgsl!("shaders/compute.wgsl"));

        let compute_bind_group_layout: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let compute_bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&id_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let compute_pipeline_layout: wgpu::PipelineLayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&accumulate_bgl, &velocity_bgl, &compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline: wgpu::ComputePipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,

            accumulate_pipeline,
            velocity_pipeline,
            compute_pipeline,

            accumulate_bind_group,
            velocity_bind_group,
            compute_bind_group,

            input_texture,
            accumulation_texture,
            output_texture,
            position_texture,
            id_texture,

            render_bundle,

            viewport_origin: glam::Vec2::ZERO,
            viewport_size: glam::Vec2::new(size.width as f32, size.height as f32),
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>)
    {
        if new_size.width > 0 && new_size.height > 0
        {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }

        let min: f32 = (self.config.width as f32).min(self.config.height as f32 * ASPECT_RATIO);

        let width: f32 = self.config.width as f32 * (min / self.config.width as f32);
        let height: f32 = self.config.height as f32 * (min / (self.config.height as f32 * ASPECT_RATIO));

        self.viewport_origin = glam::Vec2::new(self.config.width as f32 - width, self.config.height as f32 - height) / 2.0;
        self.viewport_size = glam::Vec2::new(width, height);
    }

    pub(crate) fn update(
        &mut self,
        data: &[glam::Vec4],
        position: &[glam::Vec4],
        id: &[u32],
        inv_projection: &glam::Mat4,
        last_inv_projection: &glam::Mat4,
    )
    {
        let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Update Encoder"),
        });

        let size: wgpu::Extent3d = wgpu::Extent3d {
            width: IMAGE_WIDTH as u32,
            height: IMAGE_HEIGHT as u32,
            depth_or_array_layers: 1,
        };

        self.queue.write_texture(
            self.input_texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new((std::mem::size_of::<glam::Vec4>() * IMAGE_WIDTH) as u32),
                rows_per_image: None,
            },
            size,
        );

        self.queue.write_texture(
            self.position_texture.as_image_copy(),
            bytemuck::cast_slice(position),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new((std::mem::size_of::<glam::Vec4>() * IMAGE_WIDTH) as u32),
                rows_per_image: None,
            },
            size,
        );

        self.queue.write_texture(
            self.id_texture.as_image_copy(),
            bytemuck::cast_slice(id),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new((std::mem::size_of::<u32>() * IMAGE_WIDTH) as u32),
                rows_per_image: None,
            },
            size,
        );

        let mut compute_pass: wgpu::ComputePass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass") });

        let num_workgroups: glam::UVec2 = glam::UVec2::new((IMAGE_WIDTH as f32 / 16.0).ceil() as u32, (IMAGE_HEIGHT as f32 / 16.0).ceil() as u32);

        if inv_projection == last_inv_projection
        {
            compute_pass.set_pipeline(&self.accumulate_pipeline);
            compute_pass.set_bind_group(0, &self.accumulate_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups.x, num_workgroups.y, 1);
        }
        else
        {
            compute_pass.set_pipeline(&self.velocity_pipeline);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(last_inv_projection));
            compute_pass.set_bind_group(0, &self.velocity_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups.x, num_workgroups.y, 1);

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.accumulate_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.velocity_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups.x, num_workgroups.y, 1);
        }

        drop(compute_pass);

        encoder.copy_texture_to_texture(self.output_texture.as_image_copy(), self.accumulation_texture.as_image_copy(), size);

        self.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn save(&self)
    {
        // let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //     label: Some("Save Image Encoder"),
        // });
        //
        // encoder.copy_texture_to_buffer(
        //     self.accumulation_texture.as_image_copy(),
        //     wgpu::ImageCopyBuffer {
        //         buffer: &self.output_buffer,
        //         layout: wgpu::ImageDataLayout {
        //             offset: 0,
        //             bytes_per_row: std::num::NonZeroU32::new((4 * std::mem::size_of::<u8>() * IMAGE_WIDTH) as u32),
        //             rows_per_image: None,
        //         },
        //     },
        //     wgpu::Extent3d {
        //         width: IMAGE_WIDTH as u32,
        //         height: IMAGE_HEIGHT as u32,
        //         depth_or_array_layers: 1,
        //     },
        // );
        //
        // self.queue.submit(Some(encoder.finish()));
        //
        // let read_buffer = wgpu::util::DownloadBuffer::read_buffer(&self.device, &self.queue, &self.output_buffer.slice(..));
        // self.device.poll(wgpu::Maintain::Wait);
        // let data: wgpu::util::DownloadBuffer = read_buffer.block_on().unwrap();
        //
        // image::save_buffer(
        //     "images/output_test.png",
        //     bytemuck::cast_slice(&data),
        //     IMAGE_WIDTH as u32,
        //     IMAGE_HEIGHT as u32,
        //     image::ColorType::Rgba8,
        // )
        // .unwrap();
        // self.output_buffer.unmap();
    }

    // https://docs.rs/wgpu/latest/wgpu/util/struct.DownloadBuffer.html
    pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError>
    {
        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view: wgpu::TextureView = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let mut render_pass: wgpu::RenderPass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_viewport(
            self.viewport_origin.x,
            self.viewport_origin.y,
            self.viewport_size.x,
            self.viewport_size.y,
            0.0,
            1.0,
        );
        render_pass.execute_bundles(std::iter::once(&self.render_bundle));

        drop(render_pass);

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}
