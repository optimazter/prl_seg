function rotmat = quat_to_rot(quatern_a, quatern_b, quatern_c, quatern_d)

    %Transform from quaternion representation to rotation matrix
    Rxx = 1 - 2*(quatern_c^2 + quatern_d^2);
    Rxy = 2*(quatern_b*quatern_c - quatern_d*quatern_a);
    Rxz = 2*(quatern_b*quatern_d + quatern_c*quatern_a);

    Ryx = 2*(quatern_b*quatern_c + quatern_d*quatern_a);
    Ryy = 1 - 2*(quatern_b^2 + quatern_d^2);
    Ryz = 2*(quatern_c*quatern_d - quatern_b*quatern_a );

    Rzx = 2*(quatern_b*quatern_d - quatern_c*quatern_a );
    Rzy = 2*(quatern_c*quatern_d + quatern_b*quatern_a );
    Rzz = 1 - 2 *(quatern_b^2 + quatern_c^2);

    rotmat = [ 
        Rxx,    Rxy,    Rxz;
        Ryx,    Ryy,    Ryz;
        Rzx,    Rzy,    Rzz];
end