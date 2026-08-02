from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    build_teacher_target_artifact,
    derive_teacher_view,
)


def _generations() -> tuple[TeacherGeneration, ...]:
    clusters = ("a", "a", "a", "b", "b", "c", "d", "e")
    qualities = (0.2, 0.9, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3)
    return tuple(
        TeacherGeneration(
            prompt_id="paper-1",
            sample_index=index,
            text=f"sample-{index}",
            quality_score=qualities[index],
            cluster_id=clusters[index],
        )
        for index in range(8)
    )


def test_teacher_views_are_deterministic_and_follow_declared_selection() -> None:
    generations = _generations()

    random_first = derive_teacher_view(generations, view="random1", seed=17)
    random_again = derive_teacher_view(tuple(reversed(generations)), view="random1", seed=17)
    best = derive_teacher_view(generations, view="best1", seed=17)
    mode = derive_teacher_view(generations, view="mode1", seed=17)
    diverse = derive_teacher_view(generations, view="diverse4", seed=17)

    assert random_first == random_again
    assert len(random_first) == 1
    assert best[0].sample_index == 1
    assert mode[0].sample_index == 1
    assert len(diverse) == 4
    assert len({generation.cluster_id for generation in diverse}) == 4
    assert diverse[0].sample_index == 1


def test_teacher_target_artifact_derives_every_training_view_once() -> None:
    artifact = build_teacher_target_artifact(_generations(), seed=17)

    assert artifact["schema_version"] == 1
    views = artifact["targets"]["paper-1"]
    assert set(views) == {"random1", "best1", "mode1", "diverse4", "all8"}
    assert len(views["random1"]) == len(views["best1"]) == len(views["mode1"]) == 1
    assert len(views["diverse4"]) == 4
    assert len(views["all8"]) == 8
