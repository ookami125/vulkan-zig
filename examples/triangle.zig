const std = @import("std");
const vk = @import("vk.zig");
const c = @import("c.zig");
const glm = @import("zlm.zig");
const shaders = @import("shaders");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;

const app_name = "vulkan-zig triangle example";

const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
    };

    pos: glm.Vec3,
    color: glm.Vec3,
};

const vertices = [_]Vertex{
    .{ .pos = .{ .x=-1, .y=-1, .z= 1, }, .color = .{ .x=1.0, .y=1.0, .z=0.0 } },
    .{ .pos = .{ .x= 1, .y=-1, .z= 1, }, .color = .{ .x=0.0, .y=1.0, .z=0.0 } },
    .{ .pos = .{ .x=-1, .y= 1, .z= 1, }, .color = .{ .x=1.0, .y=0.0, .z=0.0 } },
    .{ .pos = .{ .x= 1, .y= 1, .z= 1, }, .color = .{ .x=0.0, .y=0.0, .z=0.0 } },
    .{ .pos = .{ .x=-1, .y=-1, .z=-1, }, .color = .{ .x=1.0, .y=1.0, .z=1.0 } },
    .{ .pos = .{ .x= 1, .y=-1, .z=-1, }, .color = .{ .x=0.0, .y=1.0, .z=1.0 } },
    .{ .pos = .{ .x=-1, .y= 1, .z=-1, }, .color = .{ .x=1.0, .y=0.0, .z=1.0 } },
    .{ .pos = .{ .x= 1, .y= 1, .z=-1, }, .color = .{ .x=0.0, .y=0.0, .z=1.0 } },
};

const indicies = [_]u16{
	2, 7, 6,
	2, 3, 7,
	0, 4, 5,
	0, 5, 1,
	0, 2, 6,
	0, 6, 4,
	1, 7, 3,
	1, 5, 7,
	0, 3, 2,
	0, 1, 3,
	4, 6, 7,
	4, 7, 5,
};

const UniformBufferObject = struct {
    model: glm.Mat4 align(16),
    view: glm.Mat4 align(16),
    proj: glm.Mat4 align(16),
};

var firstTime: i64 = 0;
pub fn timestamp() f32 {
    if(firstTime == 0) firstTime = std.time.microTimestamp();
    var diffTime = @as(f32, @floatFromInt(std.time.microTimestamp() - firstTime)) / @as(f32, std.time.us_per_s);
    return diffTime;
}

fn subtype(comptime t: type) type {
    const tt = @typeInfo(t);
    return switch(tt) {
        .Pointer => |p| p.child,
        else => error.InvalidType,
    };
}

pub fn ptr2slice(a: anytype) []subtype(@TypeOf(a)) {
    return @as([*]subtype(@TypeOf(a)), @ptrCast(@alignCast(a)))[0..1];
}

pub fn memcpy(dest: anytype, source: anytype, bytes: usize) void {
    var dest_slice: []u8 = @as([*]u8, @ptrCast(@alignCast(dest)))[0..bytes];
    var source_slice: []u8 = @as([*]u8, @ptrCast(@alignCast(source)))[0..bytes];
    std.mem.copyForwards(u8, dest_slice, source_slice);
}

pub fn main() !void {
    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    if (c.glfwVulkanSupported() != c.GLFW_TRUE) {
        std.log.err("GLFW could not find libvulkan", .{});
        return error.NoVulkan;
    }

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window = c.glfwCreateWindow(
        @intCast(extent.width),
        @intCast(extent.height),
        app_name,
        null,
        null,
    ) orelse return error.WindowInitFailed;
    defer c.glfwDestroyWindow(window);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {s}\n", .{gc.deviceName()});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    var ubo_layout_binding = vk.DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptor_count = 1,
        .descriptor_type = .uniform_buffer,
        .p_immutable_samplers = null,
        .stage_flags = .{
            .vertex_bit = true,
        }
    };

    var layout_info = vk.DescriptorSetLayoutCreateInfo{
        .binding_count = 1,
        .p_bindings = @ptrCast(&ubo_layout_binding),
    };

    var descriptor_set_layout = try gc.vkd.createDescriptorSetLayout(gc.dev, &layout_info, null);
    defer gc.vkd.destroyDescriptorSetLayout(gc.dev, descriptor_set_layout, null);

    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    };

    const pipeline_layout = try gc.vkd.createPipelineLayout(gc.dev, &pipeline_layout_info, null);
    defer gc.vkd.destroyPipelineLayout(gc.dev, pipeline_layout, null);

    const render_pass = try createRenderPass(&gc, swapchain);
    defer gc.vkd.destroyRenderPass(gc.dev, render_pass, null);

    var pipeline = try createPipeline(&gc, pipeline_layout, render_pass);
    defer gc.vkd.destroyPipeline(gc.dev, pipeline, null);

    var framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain);
    defer destroyFramebuffers(&gc, allocator, framebuffers);

    const pool = try gc.vkd.createCommandPool(gc.dev, &.{
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.vkd.destroyCommandPool(gc.dev, pool, null);

    const vertex_buffer = try gc.vkd.createBuffer(gc.dev, &.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.vkd.destroyBuffer(gc.dev, vertex_buffer, null);
    const vertex_buffer_mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, vertex_buffer);
    const vertex_buffer_memory = try gc.allocate(vertex_buffer_mem_reqs, .{ .device_local_bit = true });
    defer gc.vkd.freeMemory(gc.dev, vertex_buffer_memory, null);
    try gc.vkd.bindBufferMemory(gc.dev, vertex_buffer, vertex_buffer_memory, 0);

    const index_buffer = try gc.vkd.createBuffer(gc.dev, &.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .index_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.vkd.destroyBuffer(gc.dev, index_buffer, null);
    const index_buffer_mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, index_buffer);
    const index_buffer_memory = try gc.allocate(index_buffer_mem_reqs, .{ .device_local_bit = true });
    defer gc.vkd.freeMemory(gc.dev, index_buffer_memory, null);
    try gc.vkd.bindBufferMemory(gc.dev, index_buffer, index_buffer_memory, 0);

    try uploadToGPUBuffer(&gc, pool, vertex_buffer, vertices);
    try uploadToGPUBuffer(&gc, pool, index_buffer, indicies);

    var bufferSize: vk.DeviceSize = @sizeOf(UniformBufferObject);

    const MAX_FRAMES_IN_FLIGHT: usize = 3;

    var uniformBuffers: [MAX_FRAMES_IN_FLIGHT]vk.Buffer = .{};
    var uniformBuffersMemory: [MAX_FRAMES_IN_FLIGHT]vk.DeviceMemory = .{};
    var uniformBuffersMapped: [MAX_FRAMES_IN_FLIGHT]*u8 = .{};

    for(0..MAX_FRAMES_IN_FLIGHT) |i| {
        try createBuffer(&gc, bufferSize, .{
            .uniform_buffer_bit = true,
        }, .{
            .host_visible_bit = true,
            .host_coherent_bit = true
        }, &uniformBuffers[i], &uniformBuffersMemory[i]);

        uniformBuffersMapped[i] = @ptrCast(try gc.vkd.mapMemory(gc.dev, uniformBuffersMemory[i], 0, bufferSize, .{}));
    }

    defer for(0..MAX_FRAMES_IN_FLIGHT) |i| {
        gc.vkd.freeMemory(gc.dev, uniformBuffersMemory[i], null);
        gc.vkd.destroyBuffer(gc.dev, uniformBuffers[i], null);
    };

    var pool_size = vk.DescriptorPoolSize{
        .type = .uniform_buffer,
        .descriptor_count = MAX_FRAMES_IN_FLIGHT,
    };

    var poolInfo = vk.DescriptorPoolCreateInfo{
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&pool_size),
        .max_sets = MAX_FRAMES_IN_FLIGHT,
    };
    
    var descriptor_pool = try gc.vkd.createDescriptorPool(gc.dev, &poolInfo, null);
    defer gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);

    var layouts: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout = .{};
    inline for(0..layouts.len) |i| {
        layouts[i] = descriptor_set_layout;
    }

    var allocInfo = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = MAX_FRAMES_IN_FLIGHT,
        .p_set_layouts = &layouts,
    };

    var descriptor_sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet = .{};
    try gc.vkd.allocateDescriptorSets(gc.dev, &allocInfo, &descriptor_sets);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        var buffer_info = vk.DescriptorBufferInfo{
            .buffer = uniformBuffers[i],
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };
        var descriptor_write = vk.WriteDescriptorSet{
            .dst_set = descriptor_sets[i],
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .p_buffer_info = @ptrCast(&buffer_info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };
        gc.vkd.updateDescriptorSets(gc.dev, 1, @ptrCast(&descriptor_write), 0, null);
    }

    var currentFrame: usize = 0;
    var cmdbufs = try createCommandBuffers(
        &gc,
        pool,
        allocator,
        vertex_buffer,
        index_buffer,
        pipeline_layout,
        &descriptor_sets,
        swapchain.extent,
        render_pass,
        pipeline,
        framebuffers,
        currentFrame,
    );
    defer destroyCommandBuffers(&gc, pool, allocator, cmdbufs);

	var aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
	var ubo = UniformBufferObject{
		.model = glm.Mat4.identity,
		.view = glm.lookAt(glm.Vec3.new(0.0, 0.0, 4.0), glm.Vec3.new(0.0, 0.0, 0.0), glm.Vec3.new(0.0, 1.0, 0.0)),
		.proj = glm.perspective(glm.toRadians(45.0), aspect, 0.1, 1000.0),
	};

	var cam_pos: glm.Vec3 = glm.vec3(0, 0, 5);
	var cam_pitch: f32 = 0;
	var cam_yaw: f32 = 0;
	var cam_pitch_limit: f32 = glm.toRadians(89.99); 
	
    var startTime = timestamp();
	var lastTime = startTime;
    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        const cmdbuf = cmdbufs[swapchain.image_index];

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        var w: c_int = undefined;
        var h: c_int = undefined;
        c.glfwGetWindowSize(window, &w, &h);

		var currentTime = timestamp();
		var time = (currentTime - startTime);
		var deltaTime = currentTime - lastTime;
		
		lastTime = currentTime;
        {
			ubo.model = glm.Mat4.createAngleAxis(glm.Vec3.new(0.0, 0.0, 1.0), time * glm.toRadians(90.0));
        }

		{
			const moveSpeed: f32 = 2.0;
			const rotationSpeed: f32 = 1.0;
			
			const cam_up = glm.vec3(0, 1, 0);
			const cam_true_forward = glm.vec3(@cos(cam_pitch)*@sin(cam_yaw), @sin(cam_pitch), @cos(cam_pitch)*@cos(cam_yaw)).normalize();
			const cam_right = cam_true_forward.cross(cam_up).normalize();
			const cam_forward = glm.Vec3.zero.sub(cam_right.cross(cam_up)).normalize(); 
			if(c.glfwGetKey(window, c.GLFW_KEY_W) == c.GLFW_PRESS) {
				cam_pos = cam_pos.sub(cam_forward.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_S) == c.GLFW_PRESS) {
				cam_pos = cam_pos.add(cam_forward.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_D) == c.GLFW_PRESS) {
				cam_pos = cam_pos.sub(cam_right.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_A) == c.GLFW_PRESS) {
				cam_pos = cam_pos.add(cam_right.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
				cam_pos = cam_pos.sub(cam_up.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS) {
				cam_pos = cam_pos.add(cam_up.scale(moveSpeed*deltaTime));
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_UP) == c.GLFW_PRESS) {
				cam_pitch += rotationSpeed * deltaTime;
				if(cam_pitch > cam_pitch_limit) cam_pitch = cam_pitch_limit;
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_DOWN) == c.GLFW_PRESS) {
				cam_pitch -= rotationSpeed * deltaTime;
				if(cam_pitch < -cam_pitch_limit) cam_pitch = -cam_pitch_limit;
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_LEFT) == c.GLFW_PRESS) {
				cam_yaw += rotationSpeed * deltaTime;
			}
			if(c.glfwGetKey(window, c.GLFW_KEY_RIGHT) == c.GLFW_PRESS) {
				cam_yaw -= rotationSpeed * deltaTime;
			}

			const cam_true_forward_2 = glm.vec3(@cos(cam_pitch)*@sin(cam_yaw), @sin(cam_pitch), @cos(cam_pitch)*@cos(cam_yaw));

			ubo.view = glm.Mat4.createLook(cam_pos, cam_true_forward_2, cam_up);
		}

        if (state == .suboptimal or extent.width != @as(u32, @intCast(w)) or extent.height != @as(u32, @intCast(h))) {
            extent.width = @intCast(w);
            extent.height = @intCast(h);

			try gc.vkd.queueWaitIdle(gc.graphics_queue.handle);
            try swapchain.recreate(extent);

            destroyFramebuffers(&gc, allocator, framebuffers);
            framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain);

            destroyCommandBuffers(&gc, pool, allocator, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &gc,
                pool,
                allocator,
                vertex_buffer,
                index_buffer,
                pipeline_layout,
                &descriptor_sets,
                swapchain.extent,
                render_pass,
                pipeline,
                framebuffers,
                currentFrame,
            );

			aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
			ubo.proj = glm.perspective(glm.toRadians(45.0), aspect, 0.1, 10.0);
        }

		memcpy(uniformBuffersMapped[currentFrame], &ubo, @sizeOf(UniformBufferObject));

        c.glfwPollEvents();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    try swapchain.waitForAllFences();
}

fn uploadToGPUBuffer(gc: *const GraphicsContext, pool: vk.CommandPool, buffer: vk.Buffer, data: anytype) !void {
    const data_size = @sizeOf(@TypeOf(data));
    const data_type: type = blk: {
        const typeinfo = @typeInfo(@TypeOf(data));
        break :blk switch (typeinfo) {
            .Array => |st| st.child,
            else => error.InvalidType,
        };
    };

    const staging_buffer = try gc.vkd.createBuffer(gc.dev, &.{
        .size = data_size,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.vkd.destroyBuffer(gc.dev, staging_buffer, null);
    const mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, staging_buffer);
    const staging_memory = try gc.allocate(mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer gc.vkd.freeMemory(gc.dev, staging_memory, null);
    try gc.vkd.bindBufferMemory(gc.dev, staging_buffer, staging_memory, 0);

    const memory = try gc.vkd.mapMemory(gc.dev, staging_memory, 0, vk.WHOLE_SIZE, .{});
    defer gc.vkd.unmapMemory(gc.dev, staging_memory);

    const gpu_data: [*]data_type = @ptrCast(@alignCast(memory));
    std.mem.copyForwards(data_type, gpu_data[0..data.len], &data);

    try copyBuffer(gc, pool, buffer, staging_buffer, data_size);
}

fn copyBuffer(gc: *const GraphicsContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf: vk.CommandBuffer = undefined;
    try gc.vkd.allocateCommandBuffers(gc.dev, &.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    defer gc.vkd.freeCommandBuffers(gc.dev, pool, 1, @ptrCast(&cmdbuf));

    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    gc.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast(&region));

    try gc.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .p_wait_dst_stage_mask = undefined,
    };
    try gc.vkd.queueSubmit(gc.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try gc.vkd.queueWaitIdle(gc.graphics_queue.handle);
}

fn createBuffer(gc: *const GraphicsContext, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, buffer: *vk.Buffer, buffer_memory: *vk.DeviceMemory) !void {
    var buffer_info = vk.BufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    };
    buffer.* = try gc.vkd.createBuffer(gc.dev, &buffer_info, null);
    var mem_requirements = gc.vkd.getBufferMemoryRequirements(gc.dev, buffer.*);
    buffer_memory.* = try gc.allocate(mem_requirements, properties);
    try gc.vkd.bindBufferMemory(gc.dev, buffer.*, buffer_memory.*, 0);
}

fn findMemoryType(gc: *const GraphicsContext, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
	var mem_properties = gc.vki.getPhysicalDeviceMemoryProperties(gc.pdev);
	
	for (0..mem_properties.memory_type_count) |i| {
		if ((type_filter & (@as(u32, 1) << @truncate(i)) > 0) and vk.MemoryPropertyFlags.contains(mem_properties.memory_types[i].property_flags, properties)) {
			return @truncate(i);
		}
	}

	return error.NoSuitableMemoryType;
}

fn createCommandBuffers(
    gc: *const GraphicsContext,
    pool: vk.CommandPool,
    allocator: Allocator,
    vertex_buffer: vk.Buffer,
    index_buffer: vk.Buffer,
    pipeline_layout: vk.PipelineLayout,
    descriptorSets: [*]vk.DescriptorSet,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    framebuffers: []vk.Framebuffer,
    currentFrame: usize,
) ![]vk.CommandBuffer {
    const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(cmdbufs);

    try gc.vkd.allocateCommandBuffers(gc.dev, &.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = @as(u32, @truncate(cmdbufs.len)),
    }, cmdbufs.ptr);
    errdefer gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @as(f32, @floatFromInt(extent.width)),
        .height = @as(f32, @floatFromInt(extent.height)),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    for (cmdbufs, framebuffers) |cmdbuf, framebuffer| {
        try gc.vkd.beginCommandBuffer(cmdbuf, &.{});

        gc.vkd.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));
        gc.vkd.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));

        // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        };

        gc.vkd.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = render_pass,
            .framebuffer = framebuffer,
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @as([*]const vk.ClearValue, @ptrCast(&clear)),
        }, .@"inline");

        gc.vkd.cmdBindPipeline(cmdbuf, .graphics, pipeline);
        const offset = [_]vk.DeviceSize{0};
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&vertex_buffer), &offset);
        gc.vkd.cmdBindIndexBuffer(cmdbuf, index_buffer, 0, .uint16);
        gc.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, @ptrCast(&descriptorSets[currentFrame]), 0, null);
        gc.vkd.cmdDrawIndexed(cmdbuf, indicies.len, 1, 0, 0, 0);

        gc.vkd.cmdEndRenderPass(cmdbuf);
        try gc.vkd.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(gc: *const GraphicsContext, pool: vk.CommandPool, allocator: Allocator, cmdbufs: []vk.CommandBuffer) void {
    gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

fn createFramebuffers(gc: *const GraphicsContext, allocator: Allocator, render_pass: vk.RenderPass, swapchain: Swapchain) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.vkd.createFramebuffer(gc.dev, &.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @as([*]const vk.ImageView, @ptrCast(&swapchain.swap_images[i].view)),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(framebuffers);
}

fn createRenderPass(gc: *const GraphicsContext, swapchain: Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    return try gc.vkd.createRenderPass(gc.dev, &.{
        .attachment_count = 1,
        .p_attachments = @as([*]const vk.AttachmentDescription, @ptrCast(&color_attachment)),
        .subpass_count = 1,
        .p_subpasses = @as([*]const vk.SubpassDescription, @ptrCast(&subpass)),
    }, null);
}

fn createPipeline(
    gc: *const GraphicsContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try gc.vkd.createShaderModule(gc.dev, &.{
        .code_size = shaders.triangle_vert.len,
        .p_code = @as([*]const u32, @ptrCast(&shaders.triangle_vert)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, vert, null);

    const frag = try gc.vkd.createShaderModule(gc.dev, &.{
        .code_size = shaders.triangle_frag.len,
        .p_code = @as([*]const u32, @ptrCast(&shaders.triangle_frag)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, frag, null);

    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
        },
    };

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_description.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&pcbas),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = 2,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try gc.vkd.createGraphicsPipelines(
        gc.dev,
        .null_handle,
        1,
        @ptrCast(&gpci),
        null,
        @ptrCast(&pipeline),
    );
    return pipeline;
}
