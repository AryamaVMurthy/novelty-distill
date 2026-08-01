from novelty_distill.data.teacher_views import TeacherGeneration, derive_teacher_view


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
