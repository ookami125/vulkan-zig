const std = @import("std");

pub const SpecializeOn = @import("zlm-generic.zig").SpecializeOn;

/// Converts degrees to radian
pub fn toRadians(deg: anytype) @TypeOf(deg) {
    return std.math.pi * deg / 180.0;
}

/// Converts radian to degree
pub fn toDegrees(rad: anytype) @TypeOf(rad) {
    return 180.0 * rad / std.math.pi;
}

// export all vectors by-default to f32
pub usingnamespace SpecializeOn(f32);

const Self = @This();

pub fn lookAt(eye: Self.Vec3, center: Self.Vec3, up: Self.Vec3) Self.Mat4 {
	const f: Self.Vec3 = center.sub(eye).normalize();
	const s: Self.Vec3 = f.cross(up).normalize();
	const u: Self.Vec3 = s.cross(f);

	var Result = Self.Mat4.identity;
	Result.fields[0][0] =  s.x;
	Result.fields[1][0] =  s.y;
	Result.fields[2][0] =  s.z;
	Result.fields[0][1] =  u.x;
	Result.fields[1][1] = -u.y;
	Result.fields[2][1] =  u.z;
	Result.fields[0][2] = -f.x;
	Result.fields[1][2] = -f.y;
	Result.fields[2][2] = -f.z;
	Result.fields[3][0] = -s.dot(eye);
	Result.fields[3][1] = -u.dot(eye);
	Result.fields[3][2] =  f.dot(eye);
	return Result;
}

pub fn perspective(fovy: f32, aspect: f32, zNear: f32, zFar: f32) Self.Mat4 {
	var tanHalfFovy: f32 = @tan(fovy / 2.0);

	var Result = Self.Mat4.zero;
	Result.fields[0][0] = 1.0 / (aspect * tanHalfFovy);
	Result.fields[1][1] = 1.0 / (tanHalfFovy);
	Result.fields[2][2] = - (zFar + zNear) / (zFar - zNear);
	Result.fields[2][3] = -1.0;
	Result.fields[3][2] = - (2.0 * zFar * zNear) / (zFar - zNear);
	return Result;
}