using Distributed
using Test
addprocs(2)

@everywhere begin

using Distributed, Dagger, DaggerGPU
using CUDA, AMDGPU, KernelAbstractions

function myfunc(X)
    @assert !(X isa Array)
    X
end

end

function generate_thunks()
    as = [delayed(x->x+1)(1) for i in 1:10]
    delayed((xs...)->[sum(xs)])(as...)
end

@kernel function fill_kernel(A, x)
    idx = @index(Global, Linear)
    A[idx] = x
end

@test DaggerGPU.cancompute(:CUDA) || DaggerGPU.cancompute(:ROC)

@testset "CUDA" begin
    if !DaggerGPU.cancompute(:CUDA)
        @warn "No CUDA devices available, skipping tests"
    else
        cuproc = DaggerGPU.processor(:CUDA)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proctypes=[cuproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20

        @testset "KernelAbstractions" begin
            cuproc = DaggerGPU.processor(:CUDA)
            opts = Dagger.Sch.ThunkOptions(;proctypes=[cuproc])
            k = fill_kernel(CUDADevice(), 8)
            A = CuArray{Float32}(undef, 8)
            _A = collect(delayed(k)(A, 2.3); options=opts)
            @test all(_A .== 2.3)
        end
    end
end

@testset "ROCm" begin
    if !DaggerGPU.cancompute(:ROC)
        @warn "No ROCm devices available, skipping tests"
    else
        rocproc = DaggerGPU.processor(:ROC)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proctypes=[rocproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20

        @test_skip "KernelAbstractions"
        #= FIXME
        @testset "KernelAbstractions" begin
            rocproc = DaggerGPU.processor(:ROC)
            opts = Dagger.Sch.ThunkOptions(;proctypes=[rocproc])
            k = fill_kernel(ROCDevice(), 8)
            A = ROCArray{Float32}(undef, 8)
            _A = collect(delayed(k)(A, 2.3); options=opts)
            @test all(_A .== 2.3)
        end
        =#
    end
end
