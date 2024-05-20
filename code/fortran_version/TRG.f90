program Z_TRG
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: D = 2
    integer :: i, j, k, l, m, r, u, l2, d, n, a, b, g, w, D_new, no_iter, Dcut
    real(dp) :: K
    real(dp), allocatable :: T(:,:,:,:), T_new(:,:,:,:), Ma(:,:), Mb(:,:), &
                             S1(:,:,:), S2(:,:,:), S3(:,:,:), S4(:,:,:), U(:,:), V(:,:), S(:)

    no_iter = 3
    Dcut = 100
    K = 0.1_dp

    allocate(T(D,D,D,D))
    allocate(Ma(D**2,D**2))
    allocate(Mb(D**2,D**2))

    do r = 0, D-1
        do u = 0, D-1
            do l2 = 0, D-1
                do d = 0, D-1
                    T(r+1,u+1,l2+1,d+1) = 0.5_dp * (1.0_dp + (2*r-1) * (2*u-1) * (2*l2-1) * (2*d-1)) * &
                                           exp(2.0_dp * K * (r + u + l2 + d - 2))
                end do
            end do
        end do
    end do

    do n = 1, no_iter
        D_new = min(D**2, Dcut)
        allocate(S1(D,D,D_new))
        allocate(S2(D,D,D_new))
        allocate(S3(D,D,D_new))
        allocate(S4(D,D,D_new))

        Ma = 0.0_dp
        Mb = 0.0_dp

        do r = 0, D-1
            do u = 0, D-1
                do l2 = 0, D-1
                    do d = 0, D-1
                        Ma(l2 + 1 + D*u, r + 1 + D*d) = T(r+1, u+1, l2+1, d+1)
                        Mb(l2 + 1 + D*d, r + 1 + D*u) = T(r+1, u+1, l2+1, d+1)
                    end do
                end do
            end do
        end do

        call SVD_decomposition(Ma, U, S, V, D**2, D_new)
        do x = 0, D-1
            do y = 0, D-1
                do m = 0, D_new-1
                    S1(x+1, y+1, m+1) = sqrt(S(m+1)) * U(x + 1 + D*y, m+1)
                    S3(x+1, y+1, m+1) = sqrt(S(m+1)) * V(m+1, x + 1 + D*y)
                end do
            end do
        end do

        call SVD_decomposition(Mb, U, S, V, D**2, D_new)
        do x = 0, D-1
            do y = 0, D-1
                do m = 0, D_new-1
                    S2(x+1, y+1, m+1) = sqrt(S(m+1)) * U(x + 1 + D*y, m+1)
                    S4(x+1, y+1, m+1) = sqrt(S(m+1)) * V(m+1, x + 1 + D*y)
                end do
            end do
        end do

        allocate(T_new(D_new,D_new,D_new,D_new))
        T_new = 0.0_dp
        do r = 0, D_new-1
            do u = 0, D_new-1
                do l2 = 0, D_new-1
                    do d = 0, D_new-1
                        do a = 0, D-1
                            do b = 0, D-1
                                do g = 0, D-1
                                    do w = 0, D-1
                                        T_new(r+1,u+1,l2+1,d+1) = T_new(r+1,u+1,l2+1,d+1) + &
                                                                  S1(w+1,a+1,r+1) * S2(a+1,b+1,u+1) * &
                                                                  S3(b+1,g+1,l2+1) * S4(g+1,w+1,d+1)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do

        deallocate(T)
        T = T_new
        D = D_new
        deallocate(S1, S2, S3, S4)
    end do

    real(dp) :: Z
    Z = 0.0_dp
    do r = 0, D-1
        do u = 0, D-1
            do l2 = 0, D-1
                do d = 0, D-1
                    Z = Z + T(r+1, u+1, l2+1, d+1)
                end do
            end do
        end do
    end do

    print *, "Z =", Z

contains

    subroutine SVD_decomposition(A, U, S, V, n, D_new)
        implicit none
        integer, intent(in) :: n, D_new
        real(dp), intent(in) :: A(n, n)
        real(dp), intent(out) :: U(n, D_new), S(D_new), V(D_new, n)
        real(dp), allocatable :: full_U(:,:), full_V(:,:), full_S(:)
        integer :: info

        allocate(full_U(n, n))
        allocate(full_V(n, n))
        allocate(full_S(n))

        call dgesvd('A', 'A', n, n, A, n, full_S, full_U, n, full_V, n, work, lwork, info)

        S = full_S(1:D_new)
        U = full_U(:, 1:D_new)
        V = transpose(full_V(:, 1:D_new))

        deallocate(full_U, full_V, full_S)
    end subroutine SVD_decomposition

end program Z_TRG
