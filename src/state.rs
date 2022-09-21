use wgpu::include_wgsl;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::window::{Window, WindowId};

use crate::{ASPECT_RATIO, IMAGE_HEIGHT, IMAGE_WIDTH};

struct Texture
{
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Texture
{
    pub fn new(device: &wgpu::Device, size: wgpu::Extent3d, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self
    {
        let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
        });

        let view: wgpu::TextureView = texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        Self { texture, view, sampler }
    }
}

pub struct State
{
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub(crate) size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    texture: Texture,
    bind_group: wgpu::BindGroup,
    // output_texture: Texture,
    // output_buffer: wgpu::Buffer,
    // https://github.com/gfx-rs/wgpu/issues/2683
    // pub(crate) num_samples: Box<u32>,
    // render_bundle: wgpu::RenderBundle,
    pub num_samples: u32,
    pub reset: bool,
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
                        max_push_constant_size: 4,
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
        };

        surface.configure(&device, &config);

        let texture: Texture = Texture::new(
            &device,
            wgpu::Extent3d {
                width: IMAGE_WIDTH as u32,
                height: IMAGE_HEIGHT as u32,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        );

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
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        let shader: wgpu::ShaderModule = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let render_pipeline_layout: wgpu::PipelineLayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::FRAGMENT,
                range: 0..4,
            }],
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            multiview: None,
        });

        let num_samples: u32 = 0;
        let reset: bool = false;

        // https://github.com/gfx-rs/wgpu/issues/2683
        // let num_samples: Box<u32> = Box::new(0);
        //
        // let mut encoder: wgpu::RenderBundleEncoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
        //     label: None,
        //     color_formats: &[Some(config.format)],
        //     depth_stencil: None,
        //     sample_count: 1,
        //     multiview: None,
        // });
        //
        // encoder.set_pipeline(&render_pipeline);
        // encoder.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(num_samples.as_ref()));
        // encoder.set_bind_group(0, &bind_group, &[]);
        // encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        // encoder.draw(0..4, 0..1);
        //
        // let render_bundle: wgpu::RenderBundle = encoder.finish(&wgpu::RenderBundleDescriptor::default());

        // let output_buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Output Buffer"),
        //     size: (4 * std::mem::size_of::<u8>() * IMAGE_WIDTH * IMAGE_HEIGHT) as wgpu::BufferAddress,
        //     usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        //     mapped_at_creation: false,
        // });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            // vertex_buffer,
            texture,
            bind_group,
            // num_samples,
            // render_bundle,
            // output_texture,
            // output_buffer,
            num_samples,
            reset,
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
    }

    pub(crate) fn input(&mut self, event: &Event<()>, target_window_id: &WindowId) -> bool
    {
        match event
        {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { .. },
                ..
            } =>
            {
                self.reset = true;
                true
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::W | VirtualKeyCode::A | VirtualKeyCode::S | VirtualKeyCode::D),
                                ..
                            },
                        ..
                    },
                window_id,
            } if window_id == target_window_id =>
            {
                self.reset = true;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn update(&mut self, data: &[glam::Vec4])
    {
        self.num_samples = if self.reset { 1 } else { self.num_samples + 1 };
        self.reset = false;

        self.queue.write_texture(
            self.texture.texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new((std::mem::size_of::<glam::Vec4>() * IMAGE_WIDTH) as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: IMAGE_WIDTH as u32,
                height: IMAGE_HEIGHT as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn save(&self)
    {
        // let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //     label: Some("Save Image Encoder"),
        // });
        //
        // encoder.copy_texture_to_buffer(
        //     self.texture.texture.as_image_copy(),
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

        // https://github.com/gfx-rs/wgpu/issues/2683
        // render_pass.execute_bundles(std::iter::once(&self.render_bundle));

        let min: f32 = (self.config.width as f32).min(self.config.height as f32 * ASPECT_RATIO);

        let width: f32 = self.config.width as f32 * (min / self.config.width as f32);
        let height: f32 = self.config.height as f32 * (min / (self.config.height as f32 * ASPECT_RATIO));

        let x: u32 = (self.config.width - width as u32) / 2;
        let y: u32 = (self.config.height - height as u32) / 2;

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_viewport(x as f32, y as f32, width, height, 0.0, 1.0);
        render_pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&(self.num_samples as f32)));
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        drop(render_pass);

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}
